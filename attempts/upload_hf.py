from huggingface_hub import upload_folder

upload_folder(
    repo_id="Jess437/Llama-3.2-1B-Instruct-distill-gptq-b8g128",  # 目標 repo ID
    folder_path="./Llama-3.2-1B-Instruct-distill-gptq-b8g128",  # 量化模型儲存的資料夾
    path_in_repo=".",        # 上傳到 repo 根目錄
    repo_type="model"
)
