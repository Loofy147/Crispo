import gradio as gr
import os
from crispo.crispo import Crispo
from crispo.crispo_core import OrchestrationContext, MetaLearner, SkiRentalContext, OneMaxSearchContext

def generate_code(objective, complexity, trust_parameter, project_type, domain, is_laa):
    """
    Runs the Crispo engine programmatically to generate code.
    """
    try:
        # Create dummy history files if LAA is selected, as the system requires them.
        if is_laa:
            if "ski rental" in objective.lower() and not os.path.exists("ski_rental_history.csv"):
                with open("ski_rental_history.csv", "w") as f:
                    f.write("cost,decision\\n10,rent\\n10,rent\\n100,buy\\n")
            elif "one-max" in objective.lower() and not os.path.exists("one_max_history.csv"):
                 with open("one_max_history.csv", "w") as f:
                    f.write("value\\n80\\n95\\n100\\n")

        # Initialize Crispo components
        context = OrchestrationContext(project="GradioDemo", objective=objective)
        meta_learner = MetaLearner()

        # Determine the problem context for LAA objectives
        problem_context = None
        if "ski rental" in objective.lower():
            problem_context = SkiRentalContext()
        elif "one-max" in objective.lower():
            problem_context = OneMaxSearchContext()

        crispo_instance = Crispo(context, meta_learner, problem_context=problem_context)

        # Run the orchestration
        final_scripts = crispo_instance.orchestrate(
            project_type=project_type,
            domain=domain,
            complexity=complexity,
            enable_transfer_learning=False, # Keep demo simple
            enable_nas=False,
            enable_federated_optimization=False,
            trust_parameter=trust_parameter
        )

        if not final_scripts:
            return "Failed to generate scripts. The orchestrator returned an empty list.", ""

        # Format the output for the Gradio interface
        if is_laa and len(final_scripts) == 2:
            return (
                f"# --- Generated Algorithm ---\\n\\n{final_scripts[0]}",
                f"# --- Generated Predictor ---\\n\\n{final_scripts[1]}"
            )
        elif len(final_scripts) >= 2:
            return (
                f"# --- Generated Layer 0 ---\\n\\n{final_scripts[0]}",
                f"# --- Generated Layer 1 ---\\n\\n{final_scripts[1]}"
            )
        else:
            return f"# --- Generated Script ---\\n\\n{final_scripts[0]}", ""

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", ""

# Define the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Crispo: Autonomous Algorithm Co-Design")
    gr.Markdown(
        "Welcome to the interactive demo for Crispo! "
        "Define your objective and let Crispo co-design a Learning-Augmented Algorithm (LAA) "
        "or a standard multi-layer script pipeline for you."
    )

    with gr.Row():
        with gr.Column(scale=1):
            objective = gr.Textbox(
                label="High-Level Objective",
                placeholder="e.g., Generate a learning-augmented algorithm for the ski rental problem",
                value="Generate a learning-augmented algorithm for the ski rental problem"
            )
            is_laa = gr.Checkbox(
                label="Is this a Learning-Augmented Algorithm (LAA) objective?",
                value=True
            )
            complexity = gr.Slider(
                minimum=0.1, maximum=1.0, step=0.1, value=0.5, label="Complexity"
            )
            trust_parameter = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.1, value=0.8, label="Trust Parameter (λ) for LAAs"
            )

            with gr.Accordion("Advanced Options", open=False):
                project_type = gr.Dropdown(
                    ["laa_ski_rental", "laa_one_max", "data_pipeline", "web_scraper"],
                    value="laa_ski_rental",
                    label="Project Type"
                )
                domain = gr.Dropdown(
                    ["online_algorithms", "finance", "data_engineering", "logistics"],
                    value="online_algorithms",
                    label="Problem Domain"
                )

        with gr.Column(scale=2):
            output1 = gr.Code(label="Generated Script 1 (Algorithm / Layer 0)", language="python")
            output2 = gr.Code(label="Generated Script 2 (Predictor / Layer 1)", language="python")

    generate_btn = gr.Button("Generate Code", variant="primary")
    generate_btn.click(
        fn=generate_code,
        inputs=[objective, complexity, trust_parameter, project_type, domain, is_laa],
        outputs=[output1, output2]
    )

    gr.Markdown("---")
    gr.Markdown("### Example Objectives to Try:")
    gr.Examples(
        examples=[
            ["Generate a learning-augmented algorithm for the one-max search problem", 0.7, 0.9, "laa_one_max", "online_algorithms", True],
            ["Fetch data from an API, process with pandas, and save to CSV", 0.6, 0.5, "data_pipeline", "data_engineering", False],
            ["Scrape a website for headlines and summarize them", 0.8, 0.5, "web_scraper", "data_engineering", False],
        ],
        inputs=[objective, complexity, trust_parameter, project_type, domain, is_laa],
        outputs=[output1, output2],
        fn=generate_code,
        cache_examples=False,
    )

demo.launch()
