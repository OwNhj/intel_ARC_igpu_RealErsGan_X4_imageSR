#使用了openVINO,功能仅在lunarlake上的Xe igpu进行了测试
#精度为fp16
#如果图像输入大于1920*1080会先缩放至小于1920*1080（避免图片过大导致内存不足）
#需要超分的图片放置于.py同路径下即可
#可批量处理图片


#OpenVINO was used, and the functionality was only tested on Xe igpu on Lunarlake
#Accuracy is fp16
#If the image input is larger than 1920 * 1080, it will be scaled to be smaller than 1920 * 1080 first (to avoid the image being too large and causing insufficient memory)
#Images that require super-resolution can be placed in the same path as. py
#Batch processing of images is possible
