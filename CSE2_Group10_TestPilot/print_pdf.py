import os, subprocess, shutil

base_dir = os.path.abspath('CSE2_Group10_TestPilot')
compiled_pdf = os.path.join(base_dir, 'compiled', 'main.pdf')
html_file = os.path.join(base_dir, 'compiled', 'paper_render.html')

temp_pdf = os.path.join(os.environ.get('TEMP', 'C:\\Windows\\Temp'), 'testpilot_main.pdf')
chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe'

cmd = [
    chrome_path,
    '--headless=new',
    '--disable-gpu',
    '--no-pdf-header-footer',
    f'--print-to-pdf={temp_pdf}',
    html_file
]

print('Running Chrome headless to print PDF...')
subprocess.run(cmd, check=True)

if os.path.exists(temp_pdf):
    shutil.copyfile(temp_pdf, compiled_pdf)
    os.remove(temp_pdf)
    size = os.path.getsize(compiled_pdf)
    print(f'SUCCESS: compiled/main.pdf created! File size: {size} bytes')
else:
    print('Error: Temp PDF not found.')
