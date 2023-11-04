import logging
from random import random

import numpy as np
import imgui
import jsonschema
import moderngl_window as mglw
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from array import array

from defined_pipelines import defined_pipelines
from img_stats import compute_img_stats
from imgui_logger import ImGuiHandler
from pipeline.pipeline import save_pipeline_result
from schema.extended_schema_validator import ExtendedValidator

logger = logging.getLogger()
logger.propagate = True
logger.level = logging.INFO
imgui_handler = ImGuiHandler()
logger.addHandler(imgui_handler)


class WindowEvents(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "image processing framework"
    aspect_ratio = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.window_height = 0
        self.window_width = 0
        imgui.create_context()
        self.wnd.ctx.error
        self.imgui = ModernglWindowRenderer(self.wnd)

        self.pipelines = defined_pipelines
        self.output_texture = None

        self.output_img_stats = None
        self.current_pipeline_result = None
        self.current_pipeline = None
        self.current_pipeline_idx = 0
        self.current_pipeline_step_result_idx = 0
        self.current_pipeline_step_idx = 0

        self.change_current_pipeline()

    def render(self, time: float, frametime: float):
        self.render_ui()

    def render_ui(self):

        imgui.new_frame()

        x_pipeline = 0
        y_pipeline = 0
        width_pipeline = 1 / 3 * self.window_width
        height_pipeline = self.window_height

        x_output = width_pipeline
        y_output = 0
        width_output = 2 / 3 * self.window_width
        height_output = self.window_height

        imgui.set_next_window_position(x_pipeline, y_pipeline)
        imgui.set_next_window_size(width_pipeline, height_pipeline)
        imgui.begin('Pipeline Configuration')

        imgui.begin_child('##PipelineRegion', border=True, flags=imgui.WINDOW_HORIZONTAL_SCROLLING_BAR)

        self.render_ui_pipelines()
        imgui.spacing()
        imgui.spacing()
        self.render_ui_pipeline_config()
        imgui.spacing()
        imgui.spacing()
        self.render_ui_run_pipeline()

        imgui.spacing()
        self.render_ui_log()

        imgui.end_child()

        imgui.end()

        imgui.set_next_window_position(x_output, y_output)
        imgui.set_next_window_size(width_output, height_output)
        imgui.begin("Output")

        self.render_ui_pipeline_result()

        self.render_ui_output()

        imgui.end()

        imgui.end_frame()

        imgui.render()

        self.imgui.render(imgui.get_draw_data())

    def render_ui_log(self):
        imgui.label_text('##LogLabel', 'Log')
        imgui.begin_child('##Log Region', border=True, flags=imgui.WINDOW_HORIZONTAL_SCROLLING_BAR)
        imgui.text_unformatted('\n'.join(imgui_handler.buffer))
        imgui.end_child()

    def render_ui_pipeline_result(self):
        if self.current_pipeline_result:

            clicked, self.current_pipeline_step_result_idx = imgui.listbox(
                "##PipelineResultListBox",
                self.current_pipeline_step_result_idx,
                [self.current_pipeline.steps[i][0] for i, x in enumerate(self.current_pipeline_result.step_results)]
            )
            if clicked:
                self.prepare_pipeline_result()

            imgui.same_line()

            imgui.begin_group()

            if imgui.button('Save Results as Images'):
                save_pipeline_result(self.current_pipeline, self.current_pipeline_result, 'results/')

            if self.output_img_stats:
                imgui.text_colored(f'Dimensions: {self.output_img_stats["dims"]}', 0.2, 1.,0.)
                imgui.text_colored(f'Mean:', 0.2, 1., 0.)
                imgui.same_line()
                for mean in self.output_img_stats["mean"]:
                    imgui.text_colored(f'{mean:.3f} ', 0.2, 1., 0.)
                    imgui.same_line()
                imgui.new_line()
                imgui.text_colored(f'Entropy:', 0.2, 1., 0.)
                imgui.same_line()
                for entropy in self.output_img_stats["entropy"]:
                    imgui.text_colored(f'{entropy:.3f} ', 0.2, 1., 0.)
                    imgui.same_line()

            imgui.end_group()

            if self.output_img_stats:

                imgui.begin_child('##StatsRegion', width=imgui.core.get_content_region_available()[0], height=160,
                                  border=True, flags=imgui.WINDOW_HORIZONTAL_SCROLLING_BAR)

                imgui.label_text('##Histos', 'Histograms')
                imgui.begin_group()

                for channel_histo in self.output_img_stats['histo']:

                    region = imgui.core.get_content_region_available()
                    imgui.plot_histogram('', channel_histo[0].astype(np.float32), graph_size=(256, 100))
                    imgui.same_line()

                imgui.end_group()

                imgui.end_child()

    def render_ui_pipelines(self):
        if self.pipelines:

            imgui.begin_group()

            imgui.text('Available Processing Pipelines')

            clicked, self.current_pipeline_idx = imgui.listbox(
                "##PipelinesListBox",
                self.current_pipeline_idx,
                [x.name for x in self.pipelines]
            )
            if clicked:
                self.change_current_pipeline()

            imgui.end_group()

    def render_ui_pipeline_config(self):

        imgui.text(f'Configuration for {self.pipelines[self.current_pipeline_idx].name}:')

        clicked, self.current_pipeline_step_idx = imgui.listbox(
            "##StepListBox",
            self.current_pipeline_step_idx,
            [x[0] for x in self.current_pipeline.steps]
        )

        imgui.text(f'Configuration for {self.current_pipeline.steps[self.current_pipeline_step_idx][0]}:')

        self.render_ui_step_configuration(self.current_pipeline.steps[self.current_pipeline_step_idx])

    def render_ui_step_configuration(self, pipeline_step):

        step = pipeline_step[1]
        config_schema = step.config_schema()
        config = pipeline_step[2]

        # necessary to add schema properties that are not yet present in the config.
        try:
            ExtendedValidator(schema=config_schema).validate(config)
        except jsonschema.exceptions.ValidationError as ex:
            imgui.text_colored(ex.message, 1, 0, 0)

        self.render_ui_step_configuration_object(step, config)

    def render_ui_step_configuration_object(self, step, step_config):

        def handle_int_type(property_name, property_schema, property_instance, property_parent):
            current = property_instance
            imgui.text(f'{property_name}:')
            imgui.same_line()
            changed, current = imgui.slider_int(f'##{property_name}',
                                                current,
                                                min_value=property_schema['minimum'],
                                                max_value=property_schema['maximum'])
            if changed:
                property_parent[property_name] = current

        def handle_number_type(property_name, property_schema, property_instance, property_parent):
            current = property_instance
            imgui.text(f'{property_name}:')
            imgui.same_line()
            changed, current = imgui.slider_float(f'##{property_name}',
                                                  current,
                                                  min_value=property_schema['minimum'],
                                                  max_value=property_schema['maximum'])
            if changed:
                property_parent[property_name] = current

        def handle_string_type(property_name, property_schema, property_instance, property_parent):
            imgui.text(f'{property_name}:')
            imgui.same_line()
            if 'enum' in property_schema:
                current = property_schema['enum'].index(property_instance)

                clicked, current = imgui.combo(
                    f'##{property_name}', current, property_schema['enum']
                )
                if clicked:
                    property_parent[property_name] = property_schema['enum'][current]
            else:
                current = property_instance
                changed, current = imgui.input_text(f'##{property_name}',
                                                    current,
                                                    property_schema[
                                                        'maxLength'] + 1 if 'maxLength' in property_schema else 512)
                if changed:
                    property_parent[property_name] = current

        def handle_boolean_type(property_name, property_schema, property_instance, property_parent):
            current = property_instance
            imgui.text(f'{property_name}:')
            imgui.same_line()
            clicked, current = imgui.checkbox(f'##{property_name}', current)
            if clicked:
                property_parent[property_name] = current

        def handle_array_type(property_name, property_schema, property_instance, property_parent):
            imgui.text('Cannot handle arrays.')

        def handle_object_type(object_name, object_schema, object_instance, object_parent):
            if object_name:
                if imgui.tree_node(object_name, imgui.TREE_NODE_DEFAULT_OPEN):
                    handle_properties(object_instance, object_schema)
                    imgui.tree_pop()
            else:
                handle_properties(object_instance, object_schema)

        def handle_properties(object_instance, object_schema):
            if 'properties' not in object_schema:
                return
            for property_name, property_schema in object_schema['properties'].items():
                if 'type' not in property_schema:
                    continue

                property_type_handler[property_schema['type']](
                    property_name,
                    property_schema,
                    object_instance[property_name],
                    object_instance);

        property_type_handler = {
            'string': handle_string_type,
            'integer': handle_int_type,
            'array': handle_array_type,
            'number': handle_number_type,
            'boolean': handle_boolean_type,
            'object': handle_object_type
        }

        handle_object_type(None, step.config_schema(), step_config, None)

    def render_ui_run_pipeline(self):

        imgui.begin_group()

        if imgui.button("Run Pipeline"):
            if self.current_pipeline:

                for x in self.current_pipeline.steps:
                    print(x[2])

                self.current_pipeline_result = self.current_pipeline.execute()
                self.prepare_pipeline_result()

        imgui.end_group()

    def render_ui_output(self):

        imgui.begin_child("##ResultRegion", border=True, flags=imgui.WINDOW_HORIZONTAL_SCROLLING_BAR)

        if self.output_texture:
            imgui.image(self.output_texture.glo, self.output_texture.width, self.output_texture.height)

        imgui.end_child()

    def change_current_pipeline(self):
        self.current_pipeline = self.pipelines[self.current_pipeline_idx]
        self.current_pipeline_result = None
        self.current_pipeline_step_result_idx = 0
        self.current_pipeline_step_idx = 0

        if self.output_texture:
            self.output_texture.release()
            self.output_texture = None

    def prepare_pipeline_result(self):
        if self.current_pipeline_result.successful_execution():
            img = self.current_pipeline_result.step_results[self.current_pipeline_step_result_idx].output_img

            if self.output_texture:
                self.output_texture.release()

            print(f'shape: {img.shape[1::-1]}')
            # TODO: Here is a conversion from float32 => uint8 with potential issues.
            self.output_texture = self.ctx.texture(img.shape[1::-1], img.shape[2], img.clip(0.0, 255.0).astype(np.uint8))
            self.imgui.register_texture(self.output_texture)

            self.output_img_stats = compute_img_stats(img);

    def resize(self, width: int, height: int):
        self.imgui.resize(width, height)

        self.window_height = height
        self.window_width = width

    def key_event(self, key, action, modifiers):
        self.imgui.key_event(key, action, modifiers)

    def mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)

    def mouse_drag_event(self, x, y, dx, dy):
        self.imgui.mouse_drag_event(x, y, dx, dy)

    def mouse_scroll_event(self, x_offset, y_offset):
        self.imgui.mouse_scroll_event(x_offset, y_offset)

    def mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int):
        self.imgui.mouse_release_event(x, y, button)

    def unicode_char_entered(self, char):
        self.imgui.unicode_char_entered(char)


if __name__ == '__main__':
    logging.basicConfig(filename='log.log', filemode='w', level=logging.INFO)

    mglw.run_window_config(WindowEvents)
