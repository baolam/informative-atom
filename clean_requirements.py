import re

input_file = "requirements.txt"
output_file = "clean_requirements.txt"

with open(input_file, "r") as f:
    lines = f.readlines()

cleaned = []
for line in lines:
    # Nếu là dạng file:// thì loại bỏ
    if "@ file://" in line:
        match = re.match(r"^(\S+)\s+@ file://.*", line)
        if match:
            pkg = match.group(1)
            cleaned.append(f"{pkg}\n")  # Để pip tự chọn phiên bản
    else:
        cleaned.append(line)

with open(output_file, "w") as f:
    f.writelines(cleaned)

print("✅ Đã tạo file:", output_file)
