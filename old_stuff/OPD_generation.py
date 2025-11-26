import openai
from openai import OpenAI
import pandas as pd
import os

# Use environment variable for API key - DO NOT hardcode!
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

def generate_opd_with_gpt(csv_path, api_key, output_path=None, dry_run=True):
    openai.api_key = api_key
    df = pd.read_csv(csv_path)

    operational_definition = (
        "Observability: What each party needs to perceive about the other's actions/status. "
        "Predictability: What each party needs to anticipate about the other's next actions. "
        "Directability: How each party can influence/guide the other's actions."
    )

    color_interpretation = {
        "green": ("can do it all", "can improve efficiency"),
        "yellow": ("can do it all but reliability is < 100%", "can improve reliability"),
        "orange": ("can contribute but need assistance", "assistance is required"),
        "red": ("cannot do it", "cannot provide assistance")
    }

    for idx, row in df.iterrows():
        # Construct task name from Task Object + Value
        task_object = str(row.get("Task Object", "")).strip()
        value = str(row.get("Value", "")).strip()
        task = f"{task_object} {value}".strip() if task_object and value else task_object
        
        procedure = str(row.get("Procedure", "")).strip()
        category = str(row.get("Category", "")).strip()
        task_type = str(row.get("Type", "")).strip()
        
        if not task or not procedure:
            continue

        # Skip rows that already have all required data
        existing_opd = [
            str(row.get("Observability", "")).strip(),
            str(row.get("Predictability", "")).strip(), 
            str(row.get("Directability", "")).strip()
        ]
        existing_new_cols = [
            str(row.get("Information Requirement", "")).strip(),
            str(row.get("Execution type", "")).strip(),
            str(row.get("Constraint type", "")).strip()
        ]
        
        has_opd = all(opd and opd != "nan" for opd in existing_opd)
        has_new_cols = all(col and col != "nan" for col in existing_new_cols)
        
        if has_opd and has_new_cols:
            continue

        # Interpret colors
        human_star_color = str(row.get("Human*", "")).strip().lower()
        tars_color = str(row.get("TARS", "")).strip().lower()
        tars_star_color = str(row.get("TARS*", "")).strip().lower()
        human_color = str(row.get("Human", "")).strip().lower()

        human_star_interp = color_interpretation.get(human_star_color, ("unknown", "unknown"))[0]
        tars_support_interp = color_interpretation.get(tars_color, ("unknown", "unknown"))[1]
        tars_star_interp = color_interpretation.get(tars_star_color, ("unknown", "unknown"))[0]
        human_support_interp = color_interpretation.get(human_color, ("unknown", "unknown"))[1]

        assumption_sentence = (
            f"Task allocation: Human pilot {human_star_interp} as performer, TARS {tars_support_interp} as supporter. "
            f"Conversely, TARS {tars_star_interp} as performer, Human pilot {human_support_interp} as supporter."
        )

        prompt = (
            f"Aviation Task Analysis for Cessna Citation Mustang single-pilot operations with TARS virtual copilot.\n\n"
            f"Task: {task}\n"
            f"Procedure: {procedure} ({category} - {task_type})\n"
            f"{assumption_sentence}\n\n"
            f"TARS capabilities: Avionics access (autopilot, FMS), ATC communications for emergencies, instrument monitoring (except attention getters).\n\n"
            f"Generate CONCISE requirements (max 15 words each):\n"
            f"{operational_definition}\n\n"
            f"Also provide:\n"
            f"- Information Requirement: Critical data/parameters needed for task execution\n"
            f"- Execution Type: one-off, continuous, cyclic, or monitoring\n" 
            f"- Constraint Type: hard-time, soft-time, or none\n"
            f"- Time Constraint: seconds from task onset to failure/miss (or 0 if none)\n\n"
            f"Format:\n"
            f"Observability: [concise requirement]\n"
            f"Predictability: [concise requirement]\n"
            f"Directability: [concise requirement]\n"
            f"Information Requirement: [critical parameters]\n"
            f"Execution Type: [type]\n"
            f"Constraint Type: [constraint]\n"
            f"Time Constraint: [seconds or 0]"
        )

        print(f"\n--- Processing row {idx}: {task} ---")
        print(f"Procedure: {procedure} ({category} - {task_type})")

        if dry_run:
            print(f"\n--- Prompt for row {idx} ---\n{prompt}\n")
            break  # Show only the first prompt and exit loop

        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o as GPT-5 may not be available yet
                messages=[
                    {"role": "system", "content": "You are an aviation human factors expert specializing in human-AI collaboration in cockpit environments. Provide concise, actionable requirements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            answer = response.choices[0].message.content.strip()
            print(f"Response: {answer}")

            # Parse response
            for line in answer.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    
                    if key in ["Observability", "Predictability", "Directability", 
                              "Information Requirement", "Execution Type", "Constraint Type"]:
                        df.at[idx, key] = val
                    elif key == "Time Constraint":
                        # Extract numeric value
                        try:
                            time_val = int(''.join(filter(str.isdigit, val))) if any(c.isdigit() for c in val) else 0
                            df.at[idx, "Time constraint (in s)"] = time_val
                        except:
                            df.at[idx, "Time constraint (in s)"] = 0

        except Exception as e:
            print(f"⚠️ Error on row {idx} ({task}): {e}")
            continue

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    return df

# Run the function
generate_opd_with_gpt("IA.csv", api_key=api_key, dry_run=False, output_path="IA_updated.csv")