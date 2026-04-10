import os

def get_files(folder):
    # 只取文件，不递归
    return set(f for f in os.listdir(folder)
               if os.path.isfile(os.path.join(folder, f)))

def compare_folders(folder_a, folder_b):
    files_a = get_files(folder_a)
    files_b = get_files(folder_b)

    only_a = files_a - files_b
    only_b = files_b - files_a
    common = files_a & files_b

    print("========== Summary ==========")
    print(f"Folder A: {len(files_a)} files")
    print(f"Folder B: {len(files_b)} files")
    print(f"Common:   {len(common)} files")
    print(f"Only A:   {len(only_a)} files")
    print(f"Only B:   {len(only_b)} files")

    print("\n========== Only in A ==========")
    for f in sorted(only_a):
        print(f)

    print("\n========== Only in B ==========")
    for f in sorted(only_b):
        print(f)

if __name__ == "__main__":
    folder_a = "./sam2"
    folder_b = "./sam2_old"

    compare_folders(folder_a, folder_b)
