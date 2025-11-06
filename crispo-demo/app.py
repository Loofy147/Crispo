import gradio as gr
import os
import io
import sys
import traceback
from crispo.crispo import Crispo
from crispo.crispo_core import OrchestrationContext, MetaLearner, SkiRentalContext, OneMaxSearchContext

def generate_code(objective, complexity, trust_parameter, project_type, domain, is_laa, enable_nas, enable_tl):
    """
    Runs the Crispo engine programmatically and captures its log output.
    """
    log_stream = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = log_stream

    try:
        # --- Create dummy history files if LAA is selected ---
        if is_laa:
            if "ski rental" in objective.lower() and not os.path.exists("ski_rental_history.csv"):
                with open("ski_rental_history.csv", "w") as f:
                    f.write("cost,decision\\n10,rent\\n10,rent\\n100,buy\\n")
            elif "one-max" in objective.lower() and not os.path.exists("one_max_history.csv"):
                 with open("one_max_history.csv", "w") as f:
                    f.write("value\\n80\\n95\\n100\\n")

        # --- Initialize Crispo components ---
        context = OrchestrationContext(project="GradioDemo", objective=objective)
        meta_learner = MetaLearner()

        problem_context = None
        if "ski rental" in objective.lower():
            problem_context = SkiRentalContext()
        elif "one-max" in objective.lower():
            problem_context = OneMaxSearchContext()

        crispo_instance = Crispo(context, meta_learner, problem_context=problem_context)

        # --- Run the orchestration ---
        final_scripts = crispo_instance.orchestrate(
            project_type=project_type,
            domain=domain,
            complexity=complexity,
            enable_transfer_learning=enable_tl,
            enable_nas=enable_nas,
            enable_federated_optimization=False, # Keep demo simple
            trust_parameter=trust_parameter
        )

        # --- Restore stdout and get log content ---
        sys.stdout = original_stdout
        orchestration_log = log_stream.getvalue()

        if not final_scripts:
            return "Failed to generate scripts.", "", orchestration_log

        # --- Format the output for the Gradio interface ---
        if is_laa and len(final_scripts) == 2:
            return (
                f"# --- Generated Algorithm ---\\n\\n{final_scripts[0]}",
                f"# --- Generated Predictor ---\\n\\n{final_scripts[1]}",
                orchestration_log
            )
        else:
            script1 = f"# --- Generated Layer 0 ---\\n\\n{final_scripts[0]}" if len(final_scripts) > 0 else ""
            script2 = f"# --- Generated Layer 1 ---\\n\\n{final_scripts[1]}" if len(final_scripts) > 1 else ""
            return script1, script2, orchestration_log

    except Exception:
        sys.stdout = original_stdout
        error_traceback = traceback.format_exc()
        orchestration_log = log_stream.getvalue()
        return "An unexpected error occurred.", "", f"{orchestration_log}\\n\\n--- TRACEBACK ---\\n{error_traceback}"

# --- Define the new, guided tutorial Gradio interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Crispo: The Autonomous Algorithm Co-Design Assistant")

    with gr.Accordion("What is Crispo and Why is it Valuable?", open=True):
        gr.Markdown(
            """
            **Crispo solves a difficult problem: designing high-performance algorithms for complex, real-world scenarios.**

            Traditional algorithm design is manual and time-consuming. Crispo automates this by acting as an AI assistant. You provide a high-level objective, and Crispo uses a powerful combination of **Genetic Algorithms**, **Reinforcement Learning**, and advanced techniques like **Neural Architecture Search** to co-design a solution.

            **The Key Innovation: Learning-Augmented Algorithms (LAA)**
            Crispo specializes in creating LAAs, which combine a machine learning (ML) predictor with a classical algorithm.
            - The **ML predictor** learns from historical data to make smart guesses.
            - The **classical algorithm** provides a worst-case performance guarantee.
            The result is a hybrid solution that is often far more efficient than either part alone, giving you both high performance and robust safety guarantees. **This demo lets you build one in seconds.**
            """
        )

    gr.Markdown("## 1. Define Your Goal")
    with gr.Row():
        with gr.Column(scale=2):
            objective = gr.Textbox(
                label="High-Level Objective",
                info="Describe what you want to achieve. Crispo uses this to understand your intent.",
                value="Generate a learning-augmented algorithm for the ski rental problem"
            )
            is_laa = gr.Checkbox(label="Is this a Learning-Augmented Algorithm (LAA) objective?", value=True, info="Check this if you want Crispo to co-design an ML predictor and a classical algorithm together.")

        with gr.Column(scale=1):
            project_type = gr.Dropdown(
                ["laa_ski_rental", "laa_one_max", "data_pipeline", "web_scraper"],
                value="laa_ski_rental", label="Project Type", info="Select a pre-defined template that best matches your objective."
            )
            domain = gr.Dropdown(
                ["online_algorithms", "finance", "data_engineering", "logistics"],
                value="online_algorithms", label="Problem Domain", info="What field does your problem belong to? This helps the Meta-Learner."
            )

    gr.Markdown("## 2. Tune the Co-Design Parameters")
    with gr.Accordion("Tune Orchestration Parameters", open=True):
        with gr.Row():
            complexity = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.2, label="Complexity", info="Controls the amount of 'thinking' Crispo does. Higher values mean more GA generations and RL episodes, leading to better but slower results.")
            trust_parameter = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.8, label="Trust Parameter (λ)", info="For LAAs only. How much should the algorithm trust the ML prediction? 1.0 = full trust, 0.0 = ignore the prediction.")
            enable_nas = gr.Checkbox(label="Enable Neural Architecture Search (NAS)", value=False, info="Check this to have Crispo automatically design and optimize a neural network architecture for the problem.")
            enable_tl = gr.Checkbox(label="Enable Transfer Learning", value=False, info="Check this to allow Crispo to load a pre-trained model and fine-tune it, potentially speeding up the process.")

    gr.Markdown("## 3. Generate and Analyze the Solution")
    generate_btn = gr.Button("🚀 Co-Design Solution", variant="primary")

    with gr.Tabs() as tabs:
        with gr.TabItem("📜 Generated Code", id=0):
            gr.Markdown("**This is the solution Crispo designed for you.** For LAAs, you'll see two components: the robust algorithm and the ML predictor it relies on.")
            output1 = gr.Code(label="Generated Script 1 (Algorithm / Layer 0)", language="python")
            output2 = gr.Code(label="Generated Script 2 (Predictor / Layer 1)", language="python")
        with gr.TabItem("🧠 Orchestration Log", id=1):
            gr.Markdown("**This is the real-time log of Crispo's thought process.** You can see the Genetic Algorithm evolving strategies, the Reinforcement Learning agent fine-tuning them, and the Verifier testing the results.")
            log_output = gr.Textbox(label="Live Log", lines=20, interactive=False)

    generate_btn.click(
        fn=generate_code,
        inputs=[objective, complexity, trust_parameter, project_type, domain, is_laa, enable_nas, enable_tl],
        outputs=[output1, output2, log_output]
    )

    gr.Markdown("---")
    gr.Markdown("### Don't know where to start? Try these examples!")
    gr.Examples(
        examples=[
            ["Generate a learning-augmented algorithm for the one-max search problem", 0.3, 0.9, "laa_one_max", "online_algorithms", True, False, False],
            ["Fetch data from an API, process it, and save to a CSV file", 0.2, 0.5, "data_pipeline", "data_engineering", False, True, False],
            ["Create a data pipeline that uses NAS to find an optimal prediction model", 0.4, 0.5, "data_pipeline", "data_engineering", False, True, True],
        ],
        inputs=[objective, complexity, trust_parameter, project_type, domain, is_laa, enable_nas, enable_tl],
        outputs=[output1, output2, log_output],
        fn=generate_code,
        cache_examples=False,
    )

demo.launch()
