from wia_scan import *
device = prompt_choose_device_and_connect()
press_any_key_to_continue()
pillow_image = scan_side(device=device)
filename = 'scanned.png'
pillow_image.save(filename, format='png', subsampling = 0, optimize = True, progressive = True, quality = 100)
