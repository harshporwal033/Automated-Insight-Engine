import os
import time
import json
from dotenv import load_dotenv
from utils import invoke_llm

load_dotenv()


def call_llm_with_backoff(system_prompt, user_prompt, max_retries=6):
    """
    Wrapper around invoke_llm with exponential backoff for 429 errors.
    Retry schedule: 1s, 2s, 4s, 8s, 16s, 32s (max)
    """
    delay = 1

    for attempt in range(max_retries):
        try:
            return invoke_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_id="openai/gpt-oss-120b",
                temperature=0.2,
                max_completion_tokens=2048,
                top_p=1.0,
                structured_schema=None
            )
        except Exception as e:
            err = str(e)

            if "429" in err or "Too Many Requests" in err:
                print(f"[WARN] 429 rate limit. Retry in {delay} sec...")
                time.sleep(delay)
                delay = min(delay * 2, 32)
                continue

            # other errors -> rethrow immediately
            raise

    raise RuntimeError("LLM failed after max retry attempts due to rate limits.")



def generate_llm_descriptions_for_parquet_dirs(base_dir: str = "user_data/data_meaning"):
    """
    For each parquet analysis directory, generate doc_info.txt using LLM.
    """

    # list valid parquet folders
    folders = [
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f))
    ]

    total = len(folders)
    done = 0

    for folder_name in folders:
        done += 1
        print(f"\n[{done}/{total}] Processing → {folder_name}")

        parquet_dir_path = os.path.join(base_dir, folder_name)

        txt_path = os.path.join(parquet_dir_path, "summary.txt")
        if not os.path.exists(txt_path):
            print(f"[WARN] No summary.txt found in {parquet_dir_path}, skipping.")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            summary_text = f.read()

        # Collect all images
        image_files = [
            f for f in os.listdir(parquet_dir_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Build prompt
        prompt = f"""
You are given analytical notes about a dataset. 
Use this text as the global understanding of what the dataset is about:

------------------ SUMMARY TEXT BELOW ------------------
{summary_text}
--------------------------------------------------------

Create a structured report suitable for placing directly into a PDF.

Output format:

Overall Description:
[One short paragraph summarizing the theme, relevance, and meaning of all plots]

---

Image: <image_name_1>
[One or two sentences describing the key insight from this plot]

---

Image: <image_name_2>
[description]

---

Repeat for all images.

Rules:
- Use ONLY the information present in the summary text.
- Keep each insight short, scientific, and ready to place under a plot.
"""

        print("   -> Invoking LLM...")

        try:
            llm_output = call_llm_with_backoff(
                system_prompt="You are a data analysis language model that writes precise and structured insights.",
                user_prompt=prompt
            )
        except Exception as e:
            print(f"[ERROR] LLM failed for {folder_name}: {e}")
            continue

        # Save results
        doc_info_path = os.path.join(parquet_dir_path, "doc_info.txt")
        try:
            with open(doc_info_path, "w", encoding="utf-8") as f:
                f.write(llm_output.strip())
            print(f"   -> Saved doc_info.txt successfully.")
        except Exception as e:
            print(f"[ERROR] Could not write doc_info.txt: {e}")


def main():
    print("=== LLM Report Generation Pipeline Started ===")
    generate_llm_descriptions_for_parquet_dirs()
    print("=== Completed: All parquet folders processed ===")


if __name__ == "__main__":
    main()
