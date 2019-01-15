import os
import zipfile

path = input("Specify full path to root repo directory: ")
while not os.path.exists(path):
    print("The path you input is not correct - please fix.")
    path = input("Specify full path to root repo directory: ")

folder_list = ["data","results","notebooks","scripts","charts","checkpoints","archive"]

for folder in folder_list:
    if not os.path.exists(folder):
        os.mkdir(folder)
        print(f"Creating {folder}.")
    else:
        print(f"{folder} already exists - skipping.")

data_path = path + "data/"
de = input("Do you need to download an environment? (1) Yes (2) No: ")
if int(de)==1:
    os.chdir(data_path)
    platform_selection = input("Choose platform: (1) Linux (with Vis), (2) Linux (Headless), (3) Mac, (4) Windows 32-bit, (5) Windows 64-bit")
    platform_list = ["Tennis_Linux","Tennis_Linux_NoVis","Tennis.app","Tennis_Windows_x86","Tennis_Windows_x86_64"]
    platform = platform_list[int(platform_selection)-1]
    url = f"https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/{platform}.zip"
    os.system("wget " + url)

    for file in os.listdir(data_path):
        if file.split('.')[-1]=="zip":
            fp = data_path + file
            with zipfile.ZipFile(fp,'r') as zip_ref:
                zip_ref.extractall(data_path)
            print("Environment downloaded and unzipped.")

print("Setup complete.")
