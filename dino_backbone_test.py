import torch
from torchvision import transforms
from PIL import Image
import os
from dinov2.layers.patch_embed import PatchUnEmbed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),  # Tensor 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# 이미지 로드 (로컬 파일)
image_path = "saved_image.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")

img = Image.open(image_path).convert('RGB')

# 이미지를 변환 및 배치 차원 추가
input_image = transform(img).unsqueeze(0).to(device)

# DINOv2 모델 로드 (torch 2.0.1과 호환되는 방식으로 변경)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc', trust_repo=True)  # trust_repo=True 추가
model.to(device)  # 모델을 GPU로 이동
print(model)
# 모델을 평가 모드로 설정
model.eval()

# 백본 추출
backbone = model.backbone
patch_unembed = PatchUnEmbed(
    img_size  = (224, 224)
    patch_size = 16,
    embed_dim: int = 768
    )
# 백본에 입력 이미지 전달
with torch.no_grad():
    print(f"input : {input_image.shape}")
    
    # 1. 패치 임베딩
    x = backbone.patch_embed(input_image)
    print(f"patch_embed : {x.shape}")
    
    # 2. 위치 인코딩 추가 (있는 경우)
    if hasattr(backbone, 'pos_embed') and backbone.pos_embed is not None:
        x = x + backbone.pos_embed
        print(f"After pos_embed : {x.shape}")
    
    # 3. 클래스 토큰 추가 (있는 경우)
    if hasattr(backbone, 'cls_token') and backbone.cls_token is not None:
        cls_token = backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        print(f"After cls_token : {x.shape}")
    
    # 4. 블록 통과 - 각 블록의 출력을 확인
    outputs = []
    for i, blk in enumerate(backbone.blocks):
        x = blk(x)  # 각 블록을 거친 후
        outputs.append(x.clone())  # 각 블록의 출력을 저장
    
    # 블록 출력 형태 확인
    for i, output in enumerate(outputs):
        print(f"Block {i+1} output shape: {output.shape}")
    
    # 5. 정규화
    x = backbone.norm(x)
    print(f"norm : {x.shape}")
    
    # 6. 헤드 통과
    features = backbone.head(x)
    print(f"head : {features.shape}")
    
    # 7. 특징 벡터 준비 (linear_head 입력용)
    # 모든 패치 임베딩을 평탄화
    B, N, C = features.shape  # 배치, 패치 수, 채널
    features = patch_unembed(features)
    print(features.shape)
    features_flat = features.reshape(B, -1)  # [B, N*C]
    print(f"Flattened features : {features_flat.shape}")
    
    # 8. 선형 헤드 (있는 경우)
    if hasattr(model, 'linear_head'):
        if features_flat.shape[1] != model.linear_head.weight.shape[1]:
            print(f"경고: 특징 차원 ({features_flat.shape[1]})이 linear_head 입력 차원 ({model.linear_head.weight.shape[1]})과 일치하지 않습니다.")
            # 필요한 경우 차원 적응
            if features_flat.shape[1] > model.linear_head.weight.shape[1]:
                print("특징을 잘라내어 맞춥니다.")
                features_flat = features_flat[:, :model.linear_head.weight.shape[1]]
            else:
                print("특징을 확장하여 맞춥니다.")
                pad_size = model.linear_head.weight.shape[1] - features_flat.shape[1]
                features_flat = torch.nn.functional.pad(features_flat, (0, pad_size), "constant", 0)
            print(f"Adjusted features : {features_flat.shape}")
        
        predictions = model.linear_head(features_flat)
        print(f"linear_head : {predictions.shape}")



