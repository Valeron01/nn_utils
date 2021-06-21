def calculate_mask(image, t=128/255):
    h, w, c = np.where(image > t)
    mask = np.zeros_like(image)
    mask[h, w, c] = 1
    return mask

def add_watermark(image, watermark, x=None, y=None, alpha=0.9):
    image = np.copy(image)
    beta = 1 - alpha
    
    
    if y is None:
        y = np.random.randint(len(image) - len(watermark))
    
    if x is None:
        x = np.random.randint(len(image[0]) - len(watermark[0]))
        
    x = int(x)
    y = int(y)
    
    part = image[y:y+len(watermark), x:x+len(watermark[0])]
    
    wm_mask = calculate_mask(watermark)
    
    non_masked = part * (1 - wm_mask)
    
    to_watermark = part * wm_mask
    
    to_watermark = watermark * beta + to_watermark * alpha
    
    image[y:y+len(watermark), x:x+len(watermark[0])] = non_masked + to_watermark
    image = np.minimum(1, image)
    
    return image

def scale_percentage(image, p):
    h, w = image.shape[:2]
    
    h = int(h * p)
    w = int(w * p)
    
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)

def scale_to_width(image, target_width):
    w = image.shape[1]
    
    percentage = target_width / w
    return scale_percentage(image, percentage)