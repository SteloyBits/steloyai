import gradio as gr
from ..models.hub_interface import HuggingFaceHubInterface
from ..tasks.text_generation import TextGenerator
from ..tasks.image_generation import ImageGenerator

class GradioInterface:
    def __init__(self):
        self.hub = HuggingFaceHubInterface()
        self.text_generator = TextGenerator(self.hub)
        self.image_generator = ImageGenerator(self.hub)
        
    def create_interface(self):
        """Create the Gradio interface."""
        
        with gr.Blocks(title="AI Unified Platform") as app:
            gr.Markdown("# AI Unified Platform")
            gr.Markdown("Your one-stop solution for AI tasks")
            
            with gr.Tab("Text Generation"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(label="Prompt", lines=4)
                        model_dropdown = gr.Dropdown(
                            choices=self.hub.task_to_best_models["text-generation"],
                            label="Model",
                            value=self.hub.get_best_model_for_task("text-generation")
                        )
                        max_length = gr.Slider(minimum=10, maximum=1000, value=256, label="Max Length")
                        temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, label="Temperature")
                        generate_btn = gr.Button("Generate Text")
                    
                    with gr.Column():
                        text_output = gr.Textbox(label="Generated Text", lines=10)
                
                generate_btn.click(
                    fn=self.text_generator.generate,
                    inputs=[text_input, model_dropdown, max_length, temperature],
                    outputs=text_output
                )
            
            with gr.Tab("Image Generation"):
                with gr.Row():
                    with gr.Column():
                        image_prompt = gr.Textbox(label="Image Prompt", lines=3)
                        img_model_dropdown = gr.Dropdown(
                            choices=self.hub.task_to_best_models["image-generation"],
                            label="Model",
                            value=self.hub.get_best_model_for_task("image-generation")
                        )
                        steps = gr.Slider(minimum=5, maximum=100, value=50, label="Diffusion Steps")
                        guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, label="Guidance Scale")
                        img_generate_btn = gr.Button("Generate Image")
                    
                    with gr.Column():
                        image_output = gr.Image(label="Generated Image")
                
                img_generate_btn.click(
                    fn=self.image_generator.generate,
                    inputs=[image_prompt, img_model_dropdown, steps, guidance_scale],
                    outputs=image_output
                )
                
            # Add more tabs for other tasks...
        
        return app
    
    def launch(self, share=False, debug=False):
        """Launch the Gradio interface."""
        app = self.create_interface()
        app.launch(share=share, debug=debug)


if __name__ == "__main__":
    # For direct testing of the Gradio interface
    interface = GradioInterface()
    interface.launch(debug=True)
