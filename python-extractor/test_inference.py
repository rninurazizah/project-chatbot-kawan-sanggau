from paddleocr import PPStructureV3

print("Membuat pipeline...", flush=True)

pipeline = PPStructureV3()

print("Pipeline berhasil dibuat.", flush=True)

image_path = "parser_test/page_30.png"

print("Mulai inferensi...", flush=True)

output = pipeline.predict(input=image_path)

print("Inferensi selesai.", flush=True)
print("Tipe output:", type(output), flush=True)

for i, result in enumerate(output):
    print("Hasil:", i + 1, flush=True)

    result.save_to_json(
        save_path=f"parser_test/test_{i + 1}.json"
    )

    result.save_to_markdown(
        save_path=f"parser_test/test_{i + 1}.md"
    )

print("Pengujian selesai.", flush=True)