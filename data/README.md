# VietJobs data

Public source: https://huggingface.co/datasets/dinhieufam/VietJobs

The notebook downloads `VietJobs.csv` directly, without a paid API or API key. It also accepts an existing local `VietJobs.csv.gz` or CSV upload from Google Colab. Actual dataset bytes are not bundled in this project.

`retrieval_gold_template.csv` is a blank annotation template. Use a comma-separated `question` and pipe-delimited `relevant_job_ids` (e.g. `abc123|def456`) per row to evaluate retrieval on manually checked relevance judgments.

Before redistributing the data, review its actual dataset license and the rights for underlying job advertisements.
