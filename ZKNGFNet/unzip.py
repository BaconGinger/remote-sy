# 导入所需模块
import py7zr
import os
import sys

def unzip_7z_file(seven_zip_file_path, extract_target_dir):
    """
    解压无密码的7z文件到指定目录
    
    参数:
        seven_zip_file_path (str): 待解压的7z文件路径（绝对/相对路径均可）
        extract_target_dir (str): 解压后文件存放的目标目录
    返回:
        bool: 解压成功返回True，失败返回False
    """
    # 1. 检查7z文件是否存在
    if not os.path.exists(seven_zip_file_path):
        print(f"错误：未找到7z文件 -> {seven_zip_file_path}")
        return False
    
    # 2. 检查文件后缀是否为.7z（简单校验）
    if not seven_zip_file_path.lower().endswith('.7z'):
        print(f"错误：文件 {seven_zip_file_path} 不是7z格式文件")
        return False
    
    # 3. 创建目标解压目录（不存在则创建，存在则忽略）
    os.makedirs(extract_target_dir, exist_ok=True)
    
    # 4. 执行解压操作（包含异常处理）
    try:
        # 以只读模式打开7z文件，无密码则无需指定password参数
        with py7zr.SevenZipFile(seven_zip_file_path, mode='r') as seven_zip_handler:
            # 解压所有文件到目标目录
            seven_zip_handler.extractall(path=extract_target_dir)
        print(f"✅ 解压成功！")
        print(f"📁 源文件：{seven_zip_file_path}")
        print(f"📂 解压到：{os.path.abspath(extract_target_dir)}")
        return True
    
    except py7zr.Bad7zFile:
        print(f"错误：{seven_zip_file_path} 是损坏的7z文件，无法解压")
        return False
    except PermissionError:
        print(f"错误：没有权限访问 {extract_target_dir} 目录，请检查权限")
        return False
    except Exception as e:
        print(f"解压失败！未知错误：{str(e)}")
        return False

# 主函数（方便直接运行）
if __name__ == "__main__":
    # ====================== 请修改这里的路径 ======================
    # 替换为你的7z文件路径（示例："test.7z" 或 "D:/files/data.7z"）
    SOURCE_7Z_FILE = "/home/sy/lhq/data/ZKNGFNet/ckpt.7z"
    # 替换为你想解压到的目录（示例："unzip_output" 或 "E:/unzip_files"）
    TARGET_EXTRACT_DIR = "/home/sy/lhq/data/ZKNGFNet"
    # =============================================================
    
    # 调用解压函数
    success = unzip_7z_file(SOURCE_7Z_FILE, TARGET_EXTRACT_DIR)
    
    # 根据结果退出程序
    sys.exit(0 if success else 1)