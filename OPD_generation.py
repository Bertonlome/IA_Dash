import openai
from openai import OpenAI
import pandas as pd

api_key="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"

def generate_opd_with_gpt(csv_path, api_key, output_path=None, dry_run=True):
    openai.api_key = api_key
    df = pd.read_csv(csv_path)

    operational_definition = (
        "Observability refers to the degree to which the human/AI can perceive the actions, intentions, and status of the AI/human. "
        "Predictability refers to the degree to which the human/AI can anticipate what the human/AI will do next. "
        "Directability refers to the degree to which the human/AI can influence the actions or decisions of the human/AI."
    )

    color_interpretation = {
        "green": ("can do it all", "can improve efficiency"),
        "yellow": ("can do it all but reliability is < 100%", "can improve reliability"),
        "orange": ("can contribute but need assistance", "assistance is required"),
        "red": ("cannot do it", "cannot provide assistance")
    }

    for idx, row in df.iterrows():
        task = row.get("Task", "").strip()
        procedure = row.get("Procedure", "").strip()
        if not task or not procedure:
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
            f"For the task <{task}> in the procedure <{procedure}>, it has been assumed that the human pilot {human_star_interp} as a performer, "
            f"and the autonomous agent {tars_support_interp} as a supporter. Conversely, the autonomous agent {tars_star_interp} as a performer, "
            f"and the human pilot {human_support_interp} as a supporter."
        )

        prompt = (
            f"You must find Observability, Predictability and Directability requirements for the task <{task}> "
            f"used in the procedure <{procedure}>. The scenario is a business flight with a single pilot aboard a "
            f"Cessna Citation Mustang. There will be an autonomous agent acting as a virtual copilot that will be "
            f"the performer or supporter of a number of tasks shared with the single pilot. The autonomous agent has "
            f"access to avionics systems such as autopilot and FMS, can issue calls to ATC to declare simple things "
            f"such as emergency conditions, and can check the status of most controls and instruments except those "
            f"reserved for human use such as attention getters. {assumption_sentence} "
            f"The operational definition of Observability, Predictability and Directability requirements is: {operational_definition} "
            f"These requirements will be helpful to design the agent and the human-autonomy interface."
            f"\n\nPlease respond in the following format:\n"
            f"Observability: <value>\nPredictability: <value>\nDirectability: <value>"
        )

        print(f"\n--- Prompt for row {idx} ---\n{prompt}\n")

        if dry_run:
            break  # Show only the first prompt and exit loop

        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a task analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()

            for line in answer.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    if key in ["Observability", "Predictability", "Directability"]:
                        df.at[idx, key] = val

        except Exception as e:
            print(f"⚠️ Error on row {idx} ({task}): {e}")
            continue

    if output_path:
        df.to_csv(output_path, index=False)
    return df

generate_opd_with_gpt("table_data.csv", api_key=api_key, dry_run=False, output_path="table_data_with_opd.csv")