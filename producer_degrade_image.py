#producer_degrade_image.py

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv
import sys
import random
import cv2


#========================================================
""" File Tree

"where code is"
    CamVid_12_DLC_v1
        "degradation_option.csv"    -> option saved csv file
        
        original
            images                  -> where original images locate
                image_1.png
                image_2.png
                ...
        
        x4_BILINEAR                 -> where degraded images generated
            images
                image_1.png
                image_2.png
                ...




"""
#=======================================================


def imshow_pil(in_pil, **kargs):
    '''
    imshow_pil(#pil image show with plt function
               in_pil
               #(선택) (tuple) 출력 크기
              ,figsize = (,)
               #(선택) (bool) pil 이미지 정보 출력 여부 (default = True)
              ,print_info = 
              )
    '''
    
    try:
        plt.figure(figsize = kargs['figsize'])
    except:
        pass
    plt.imshow(np.array(in_pil))
    plt.show()
    
    try:
        print_info = kargs['print_info']
    except:
        print_info = True
    
    if print_info:
        try:
            print("Format:", in_pil.format, "  Mode:", in_pil.mode, "  Size (w,h):", in_pil.size)
        except:
            print("Format: No Info", "  Mode:", in_pil.mode, "  Size (w,h):", in_pil.size)
    

#=== End of imshow_pil

def load_file_path(**kargs):
    '''
    list_path_out = load_file_path(in_path_dataset = "./CamVid_12_DLC_v1"
                                  ,in_category = "original"
                                  ,in_category_sub = "images"
                                  )
    '''
    
    '''
    file path info
    
    ...
        in_path_dataset
            in_category
                in_category_sub
                    image_file.png
                    ...
                    ...
    
    
    '''
    
    
    #(str) 데이터셋 상위 경로 (PATH_BASE_IN: ./"name_dataset")
    in_path_dataset = kargs['in_path_dataset']
    
    if in_path_dataset[-1] != "/":
        in_path_dataset += "/"
    
    #(str) 데이터 종류 ("train" or "val" or "test")
    in_category = kargs['in_category']
    
    if in_category[-1] != "/":
        in_category += "/"
    
    #image, label 이미지 파일의 폴더 이름
    in_category_sub = kargs['in_category_sub']
    if in_category_sub[-1] != "/":
        in_category_sub += "/"
    
    list_name_files = os.listdir(in_path_dataset + in_category + in_category_sub)
    
    print("\nin list images: ", len(list_name_files))
    print("start with...", list_name_files[0])
    print("end with...", list_name_files[-1])
    
    #return list_name_files
    
    list_path_images = []
    for i_name in sorted(list_name_files):
        list_path_images.append(in_path_dataset + in_category + in_category_sub + i_name)
    
    return list_path_images
    
#=== End of load_file_path


def csv_2_dict(**kargs):
    '''
    dict_from_csv = csv_2_dict(path_csv = "./aaa/bbb.csv")
    '''
    
    #첫 열을 key로, 나머지 열을 value로서 list 형 묶음으로 저장한 dict 변수 생성
    #csv 파일 경로
    path_csv = kargs['path_csv']

    file_csv = open(path_csv, 'r', encoding = 'utf-8')
    lines_csv = csv.reader(file_csv)
    
    dict_csv = {}
    
    for line_csv in lines_csv:
        item_num = 0
        items_tmp = []
        for item_csv in line_csv:
            #print(item_csv)
            if item_num != 0:
                items_tmp.append(item_csv)
            item_num += 1
        dict_csv[line_csv[0]] = items_tmp
        
    file_csv.close()
    
    return dict_csv

#=== End of csv_2_dict


def degradation_total_v7(**kargs):
    func_name = "[degradation_total_v7] -->"
    #--- 변경사항
    # 
    #   1. Degradation 1회만 시행
    #   2. Degradation 결과물을 원본 크기로 복원하는 단계 시행 안함
    #      -> scale_factor 만큼 그대로 작아진 결과물 배출
    #   3. 노이즈 (Color / Gray) 생성 방식은 Real-ESRGAN 방식 그대로 적용
    #      -> Color: RGB 채널에 서로 다른 노이즈 생성 
    #      -> Gray : RGB 채널에 서로 같은 노이즈 생성
    #---
    
    '''
    사용 예시
    pil_img = degradation_total_v7(in_pil =
                                  ,is_return_options = 
                                  #--블러
                                  ,in_option_blur = "Gaussian"
                                  #-- 다운 스케일
                                  ,in_scale_factor = 
                                  ,in_option_resize = 
                                  #--노이즈
                                  ,in_option_noise = "Gaussian"
                                  #노이즈 시그마값 범위 (tuple)
                                  ,in_range_noise_sigma = 
                                  #Gray 노이즈 확룔 (int)
                                  ,in_percent_gray_noise = 
                                  #노이즈 고정값 옵션
                                  ,is_fixed_noise = 
                                  ,in_fixed_noise_channel =
                                  ,in_fixed_noise_sigma   = 
                                  )
    '''

    #IN(**):
    #       (선택, str)        in_path                 : "/..."
    #       (대체 옵션, pil)    in_pil                  : pil_img

    #고정값   (str)             in_option_blur         : "Gaussian"
    #       (선택, 2d npArray) kernel_blur            : np커널
    #
    #중지됨   (int or tuple)    in_resolution          : (사용금지) in_scale_factor로 변경됨
    #       (int or tuple)    in_scale_factor        : 스케일 팩터 (1 ~ ) -> tuple 입력시, 범위 내 균등추출 (소수점 1자리까지 사용)
    #       (str)             in_option_resize       : "AREA", "BILINEAR", "BICUBIC"
    #
    #고정값   (str)             in_option_noise        : "Gaussian"
    #       (tuple)           in_range_noise_sigma   : ((float), (float))
    #       (int)             in_percent_gray_noise  : Gray 노이즈 확률 (그 외엔 Color 노이즈로 생성), 최대 100

    #       (선택, bool)       is_fixed_noise         : 노이즈 옵션지정여부 (val & test용 , default = False)
    #       (선택, str)        in_fixed_noise_channel : 노이즈 발생 채널 지정 (val & test용) ("Color" or "Gray")
    #       (선택, str)        in_fixed_noise_sigma   : 노이즈 시그마값 지정  (val & test용)

    #       (bool)            is_return_options      : degrad- 옵션 return 여부

    #OUT(1):
    #       (PIL)             이미지
    #       (선택, str)             Degradation 옵션
    #--- --- ---

    #degrad- 옵션 return 여부
    try:
        is_return_options = kargs['is_return_options']
    except:
        is_return_options = False
    
    #(str) 사용된 degrad 옵션 저장
    return_option = ""
    
    #(str) 파일 경로 or (pil) 이미지 입력받음
    try:
        in_path = kargs['in_path']
        in_cv = cv2.imread(in_path)
    except:
        in_cv = cv2.cvtColor(np.array(kargs['in_pil']), cv2.COLOR_RGB2BGR)
    
    #입력 이미지 크기
    in_h, in_w, _ = in_cv.shape
    
    #***--- degradation_blur
    #(str) blur 방식
    in_option_blur = kargs['in_option_blur']
    
    return_option += "Blur = " + in_option_blur
    
    #평균 필터
    if in_option_blur == "Mean" or in_option_blur == "mean":
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size) / (kernel_size * kernel_size))
        out_cv_blur = cv2.filter2D(in_cv, -1, kernel)
    
    #가우시안 필터
    elif in_option_blur == "Gaussian" or in_option_blur == "gaussian":
        kernel_size = 3 #홀수만 가능
        kernel_sigma = 0.1
        kernel = cv2.getGaussianKernel(kernel_size, kernel_sigma) * cv2.getGaussianKernel(kernel_size, kernel_sigma).T
        out_cv_blur = cv2.filter2D(in_cv, -1, kernel)
    
    #기타 필터 (sinc 용)
    elif in_option_blur == "Custom" or in_option_blur == "custom":
        kernel = kargs['kernel_blur']
        out_cv_blur = cv2.filter2D(in_cv, -1, kernel)
    
    #***--- degradation_resolution
    
    #scale factor (float, 소수점 2자리 까지 사용)
    if type(kargs['in_scale_factor']) == type((0, 1)):
        in_scale_factor = round(random.uniform(kargs['in_scale_factor'][0], kargs['in_scale_factor'][-1]), 2)
    else:
        in_scale_factor = round(kargs['in_scale_factor'], 2)
    
    #최소값 clipping
    min_scale_factor = 0.25
    if in_scale_factor < min_scale_factor:
        print(func_name, "scale factor clipped to", min_scale_factor)
        in_scale_factor = min_scale_factor
    
    #(str) resize 옵션 ("AREA", "BILINEAR", "BICUBIC" / 소문자 가능)
    try:
        in_option_resize = kargs['in_option_resize']
    except:
        #default: BILINEAR
        in_option_resize = "BILINEAR"
    
    tmp_s_f = 1 / in_scale_factor
    if in_option_resize == "AREA" or in_option_resize == "area":
        tmp_interpolation = cv2.INTER_AREA
    elif in_option_resize == "BILINEAR" or in_option_resize == "bilinear":
        tmp_interpolation = cv2.INTER_LINEAR
    elif in_option_resize == "BICUBIC" or in_option_resize == "bicubic":
        tmp_interpolation = cv2.INTER_CUBIC
    
    out_cv_resize = cv2.resize(out_cv_blur, dsize=(0,0), fx=tmp_s_f, fy=tmp_s_f
                              ,interpolation = tmp_interpolation
                              )
    
    #감소된 크기 계산
    out_h, out_w, _ = out_cv_resize.shape
    
    return_option += ", Downscale(x" + str(in_scale_factor) + ") = " + in_option_resize
    
    #***--- degradation 노이즈 추가 (Color or Gray)
    
    #채널 분할
    in_cv_b, in_cv_g, in_cv_r = cv2.split(out_cv_resize)
    
    #노이즈 옵션 고정값 사용여부
    try:
        is_fixed_noise = kargs['is_fixed_noise']
    except:
        is_fixed_noise = False
    
    #노이즈 종류 선택 (Gaussian or Poisson) -> Poisson 사용 안함
    try:
        in_option_noise = kargs['in_option_noise']
    except:
        in_option_noise = "Gaussian"
    
    #노이즈 생성 (Gaussian)
    if in_option_noise == "Gaussian":
        in_noise_mu = 0 #뮤 =고정값 적용
        
        #노이즈 옵션이 지정된 경우
        if is_fixed_noise:
            #노이즈 발생 채널
            in_noise_channel = kargs['in_fixed_noise_channel']
            #시그마 값
            in_noise_sigma = int(kargs['in_fixed_noise_sigma'])
        #노이즈 옵션이 지정되지 않은 경우
        else:
            #노이즈 발생 채널 추첨
            in_percent_gray_noise = kargs['in_percent_gray_noise']
            in_noise_channel = random.choices(["Color", "Gray"]
                                             ,weights = [(100 - in_percent_gray_noise), in_percent_gray_noise]
                                             ,k = 1
                                             )[0]
            #시그마 값
            in_noise_sigma = int(random.uniform(kargs['in_range_noise_sigma'][0], kargs['in_range_noise_sigma'][-1]))
        
        #Color 노이즈 발생 (채널별 다른 노이즈 발생)
        if in_noise_channel == "Color":
            noise_r = np.random.normal(in_noise_mu, in_noise_sigma, size=(out_h, out_w))
            noise_g = np.random.normal(in_noise_mu, in_noise_sigma, size=(out_h, out_w))
            noise_b = np.random.normal(in_noise_mu, in_noise_sigma, size=(out_h, out_w))
        #Gray 노이즈 발생 (모든 채널 동일 노이즈 발생)
        elif in_noise_channel == "Gray":
            noise_r = np.random.normal(in_noise_mu, in_noise_sigma, size=(out_h, out_w))
            noise_g = noise_r
            noise_b = noise_r
        
        out_cv_r = np.uint8(np.clip(in_cv_r + noise_r, 0, 255))
        out_cv_g = np.uint8(np.clip(in_cv_g + noise_g, 0, 255))
        out_cv_b = np.uint8(np.clip(in_cv_b + noise_b, 0, 255))
    
    #채널 재 병합
    out_cv_noise = cv2.merge((out_cv_r, out_cv_g, out_cv_b))
    
    #생성옵션 기록갱신
    return_option += ", Noise = (" + in_option_noise + ", " + in_noise_channel
    return_option += ", mu = " + str(in_noise_mu) + ", sigma = " + str(in_noise_sigma) + ")"
    
    if is_return_options:
        #(PIL), (str)
        return Image.fromarray(out_cv_noise) , return_option
    else:
        #(PIL)
        return Image.fromarray(out_cv_noise)


#=== end of degradation_total_v7







def save_pil(**kargs):
    '''
    save_pil(
             #(pil) 이미지 
             pil = 
             #(str) 저장경로: ".asd/fghj"
            ,path = 
             #(str) 파일이름 + 확장자: "name.png"
            ,name = 
            )
    '''
    try:
        #저장할 pil 이미지
        in_pil = kargs['pil']
        
        #저장할 경로
        in_path = kargs['path']
        if not in_path[-1] == "/":
            in_path += "/"
        
        #파일 이름
        in_name = kargs['name']
        
        try:
            if not os.path.exists(in_path):
                os.makedirs(in_path)
                
            try:
                #이미지 저장
                in_path_name = in_path + in_name
                in_file_type = in_name.split(".")[-1]
                
                in_pil.save(in_path_name, in_file_type)
                print("PIL Image Saved: ", in_path_name)
                
            except:
                print("(except) in save_pil: save FAIL\n", in_path_name, in_file_type)
        except:
            print("(except) in save_pil: makedirs FAIL\n", in_path)
        
    except:
        print("(except) in save_pil: input error")

#=== End 0f save_pil

#******************************************************

HP_SEED = 15
HP_SCALE_FACTOR = 4
HP_INTERPOLATION_METHOD = "BILINEAR"



random.seed(HP_SEED)
np.random.seed(HP_SEED)

list_path_out = load_file_path(in_path_dataset = "./CamVid_12_DLC_v1"
                              ,in_category = "original"
                              ,in_category_sub = "images"
                              )

dict_from_csv = csv_2_dict(path_csv = "./degradation_2.csv")


for i_name in range(len(list_path_out)):
    in_path = list_path_out[i_name]
    
    in_name = in_path.split('/')[-1]
    
    try:
        csv_contents = dict_from_csv[in_name]
    except:
        csv_contents = False
        sys.exit("(exc) csv key mismatch")
    
    in_pil = Image.open(in_path)
    
    
    out_pil, out_option = degradation_total_v7(in_pil = in_pil
                                              ,is_return_options = True
                                              #--블러
                                              ,in_option_blur = "Gaussian"
                                              #-- 다운 스케일
                                              ,in_scale_factor = HP_SCALE_FACTOR
                                              ,in_option_resize = HP_INTERPOLATION_METHOD
                                              #--노이즈
                                              ,in_option_noise = "Gaussian"
                                              #노이즈 시그마값 범위 (tuple)
                                              ,in_range_noise_sigma = [1,30]
                                              #Gray 노이즈 확룔 (int)
                                              ,in_percent_gray_noise = 40
                                              #노이즈 고정값 옵션
                                              ,is_fixed_noise = True
                                              ,in_fixed_noise_channel = csv_contents[0]
                                              ,in_fixed_noise_sigma   = csv_contents[1]
                                              )
    
    print("\n\nimage file path:", in_path) #image file path: ./CamVid_12_DLC_v1/original/images/Seq05VD_f02100.png
    print("csv contents:", csv_contents) #csv contents: ['Color', '5']
    
    
    
    #중간결과 출력
    if i_name % 200 == 0:
        print("Original image")
        imshow_pil(in_pil)
        
        print("Degraded image")
        print("Option:", out_option)
        imshow_pil(out_pil)
    
    
    save_pil(
             #(pil) 이미지 
             pil = out_pil
             #(str) 저장경로: ".asd/fghj"
            ,path = "./CamVid_12_DLC_v1/x" + str(HP_SCALE_FACTOR) + "_" +HP_INTERPOLATION_METHOD + "/images"
             #(str) 파일이름 + 확장자: "name.png"
            ,name = in_name
            )
    
    
    
    if i_name == len(list_path_out) -1:
        print("\n\nTotal images:", i_name + 1)
    
    

print("End of producer_degrade_image.py")