# https://blog.csdn.net/Helloorld_1/article/details/130242871
import torch
 
import torchvision.transforms as T
 
import matplotlib.pyplot as plt
import numpy as np
 
import matplotlib.image as mpimg
 
from PIL import Image
 
from sklearn.decomposition import PCA
import matplotlib
 
matplotlib.use('TkAgg')
 
patch_h = 50
patch_w = 50
feat_dim = 1536 # vitg14
 
transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((patch_h * 14, patch_w * 14)),
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
 
dinov2_vitb14 = torch.hub.load('', 'dinov2_vitg14',source='local').cuda()
 
print(dinov2_vitb14)
 
# extract features
features = torch.zeros(4, patch_h * patch_w, feat_dim)
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14).cuda()
 
img_path = f'./image/mix.jpg'#修改图片路径
img = Image.open(img_path).convert('RGB')
imgs_tensor[0] = transform(img)[:3]
with torch.no_grad():
    features_dict = dinov2_vitb14.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']
 
 
 
features = features.reshape(4 * patch_h * patch_w, feat_dim).cpu()
print(features)
pca = PCA(n_components=3)
pca.fit(features)
pca_features = pca.transform(features)
pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (pca_features[:, 0].max() - pca_features[:, 0].min())
 
# segment using the first component
 
pca_features_fg = pca_features[:, 0] >0.3
pca_features_bg = ~pca_features_fg
 
b=np.where(pca_features_bg)#取满足条件的下标
print("1",pca_features[:, 0])
# print(pca_features_fg)
# PCA for only foreground patches
pca.fit(features[pca_features_fg])
pca_features_rem = pca.transform(features[pca_features_fg])
for i in range(3):
    pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
    # transform using mean and std, I personally found this transformation gives a better visualization
    # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5
 
pca_features_rgb = pca_features.copy()
pca_features_rgb[pca_features_fg] =pca_features_rem
pca_features_rgb[b] = 0
print("digtial",pca_features_rgb)
pca_features_rgb = pca_features_rgb.reshape(4,patch_h, patch_w, 3)
plt.imshow(pca_features_rgb[0][...,::-1])
# plt.savefig('features.png')  # 保存结果图片
plt.show()
plt.close()