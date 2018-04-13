#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#                                                                             
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Young
# DATE CREATED: 2018.04.09
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse #커맨드 라인 입력기 library
#커맨드 라인의 입력을 받아 실행되는 프로그램의 경우, 파라미터들을 파싱할 때 사용한다.
from time import time, sleep #필요한 모듈만 가져온다. 메모리를 아낄 수 있다.
from os import listdir

# Imports classifier function for using CNN to classify images 
from classifier import classifier 

# Main program function defined below
def main():
    # TODO: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time() #현재 시각
    #time()을 사용해, 시간을 측정할 수 있다.
    #1970년 1월 1일 자정 이후로 누적된 초를 float 단위로 반환
    
    # TODO: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()
    print("Command Line Arguments:\n	dir = ", in_arg.dir, 
         "\n	arch =", in_arg.arch,
         "\n	dogfile =", in_arg.dogfile)
    #add_argument()에서 추가한 인자에 접근한다.
    
    # TODO: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir) #파일이름 - 개 품종 label로 된 딕셔너리
    #이 프로젝트에선, 파일명이 해당 이미지의 품종(혹은 분류명(개가 아닐 수도 있다))를 나타낸다.
    #따라서 파일이름에서 필요한 string부분만 빼내서 정답 딕셔너리를 만든다.
    #파일명 - 정답 레이블 딕셔너리가 된다.
    
    print("\nanswers_dic has", len(answers_dic),
         "key-value pairs.\nBelow are 10 of them:")
    prnt = 0
    for key in answers_dic:
      if prnt < 10: #앞의 10개 값만 출력
        print("%2d key: %-30s label: %-26s" % (prnt+1, key, answers_dic[key]))
        #%를 사용해 포맷팅할 수 있다.
        #%d나 %s 등 앞에 숫자가 들어가면 그 숫자만큼 띄운다. 해당 포맷의 길이가 default가 된다.
        #마이너스 숫자가 들어가면 뒤로.

      prnt += 1

    # TODO: 4. Define classify_images() function to create the classifier 
    # labels with the classifier function uisng in_arg.arch, comparing the 
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch) #예측 결과 딕셔너리
    #classify_images의 반환값으로 result_dic은 각 key마다
    #index 0 : 정답. 실제 분류명(개가 아닌 것도 섞여 있다.)
    #index 1 : CNN 모델이 예측한 레이블(여러 개의 값이 될 수도 있다.) 
    #index 2 : 이미지 이름과 예측 레이블 값의 일치 여부. 0이면 Fasle(예측 맞지 않음), 1이면 True(예측 맞음).
    #를 가진다.
    
    print("\n		MATCH:")
    
    n_match = 0
    n_notmatch = 0
    
    for key in result_dic:
      if result_dic[key][2] == 1: #1인 예측 맞은 경우, 0이 틀린 경우
        n_match += 1 #예측 맞은 수 +
        print("Real : %-26s		Classifier: %-30s" % (result_dic[key][0],
                                                      result_dic[key][1]))
        #Real: 실제 이름, Classifier: CNN이 분류한 이름
    print("\n NOT A MATCH:")
    
    for key in result_dic:
      if result_dic[key][2] == 0: #1인 예측 맞은 경우, 0이 틀린 경우
        n_notmatch += 1
        print("Real: %-26s		Classifier: %-30s" % (result_dic[key][0],
                                                      result_dic[key][1]))
        #Real: 실제 이름, Classifier: CNN이 분류한 이름
    print("\n# Total Images", n_match + n_notmatch, "# Matches:", n_match, 
          "# NOT Matches:", n_notmatch)
    
    
    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile) #위에서 CNN모델로 분류한 결과에 추가한다.
    #classify_images의 반환값으로 result_dic은 각 key마다
    #index 0 : 정답. 실제 분류명(개가 아닌 것도 섞여 있다). 이미지 이름
    #index 1 : CNN 모델이 예측한 레이블(여러 개의 값이 될 수도 있다.) 
    #index 2 : 이미지 이름과 예측 레이블 값의 일치 여부. 0이면 Fasle(예측 맞지 않음), 1이면 True(예측 맞음).
    
    #이 되었다. 이것을 adjust_results4_isadog 매개변수로 전달하면, 메서드 내부를 거쳐
    #result_dic은 각 key마다 5개의 value를 가지게 된다. adjust_results4_isadog를 통해 index 3, 4가 추가된다.
    #index 0 : 정답. 실제 분류명(개가 아닌 것도 섞여 있다). 이미지 이름
    #index 1 : CNN 모델이 예측한 레이블(여러 개의 값이 될 수도 있다.) 
    #index 2 : 이미지 이름과 예측 레이블 값의 일치 여부. 0이면 Fasle(예측 맞지 않음), 1이면 True(예측 맞음).
    #index 3 : 정답. 이미지가 개인지 아닌지 여부. 0이면 Fasle(개 아님), 1이면 True(개)
    #index 4 : CNN 모델이 이미지를 개인지 아닌지 예측한 값. 0이면 Fasle(개 아님), 1이면 True(개)
    
    print("\n		MATCH:")
    
    n_match = 0
    n_notmatch = 0
    
    for key in result_dic:
      if result_dic[key][2] == 1: #이미지 이름과 예측 레이블 값의 일치한다면
        n_match += 1 #예측 맞은 수 +
        print("Real: %-26s	Classifier: %-30s	PetLabelDog: %1d  ClassLabelDog: %1d"
             % (result_dic[key][0], result_dic[key][1], result_dic[key][3], result_dic[key][4]))
        #Real: 실제 이름, Classifier: CNN이 분류한 이름, PetLabelDog: 실제 개인지 여부, ClassLabelDog: CNN이 개로 분류한지 여부
        
    print("\n NOT A MATCH: ")
    
    for key in result_dic:
      if result_dic[key][2] == 0: ##이미지 이름과 예측 레이블 값의 일치하지 않는다면
        n_notmatch += 1 #예측 틀린 수 +
        print("Real: %-26s	Classifier: %-30s	PetLabelDog: %1d  ClassLabelDog: %1d"
             % (result_dic[key][0], result_dic[key][1], result_dic[key][3], result_dic[key][4]))
        #Real: 실제 이름, Classifier: CNN이 분류한 이름, PetLabelDog: 실제 개인지 여부, ClassLabelDog: CNN이 개로 분류한지 여부
        
    print("\n# Total Images", n_match + n_notmatch, "# Matches:", n_match, 
          "# NOT Matches:", n_notmatch)
    
    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)
    
    n_images = len(result_dic) #전체 이미지 수
    n_pet_dog = 0 #개 이미지 수
    n_pet_cdog = 0 #예측 결과가 정답과 일치한 수
    n_class_cdog = 0 #이미지가 개이고, 예측도 개로 정확히 예측한 경우
    n_class_cnotd = 0 #이미지가 개가 아니면서, 예측도 개가 아닌 걸로 정확히 예측한 경우
    n_match_breed = 0 #이미지가 개면서 품종 예측까지 일치(최종 목적)
    
    #result_dic은 각 key마다 5개의 value를 가지게 된다. adjust_results4_isadog를 통해 index 3, 4가 추가된다.
    #index 0 : 정답. 실제 분류명(개가 아닌 것도 섞여 있다). 이미지 이름
    #index 1 : CNN 모델이 예측한 레이블(여러 개의 값이 될 수도 있다.) 
    #index 2 : 이미지 이름과 예측 레이블 값의 일치 여부. 0이면 Fasle(예측 맞지 않음), 1이면 True(예측 맞음).
    #index 3 : 정답. 이미지가 개인지 아닌지 여부. 0이면 Fasle(개 아님), 1이면 True(개)
    #index 4 : CNN 모델이 이미지를 개인지 아닌지 예측한 값. 0이면 Fasle(개 아님), 1이면 True(개)
    
    #calculates_results_stats의 로직과 같다.
    for key in result_dic:
      if result_dic[key][2] == 1: #이미지 이름과 예측 레이블 값이 일치하다면
        if result_dic[key][3] == 1: #이미지가 개라면
          n_pet_dog += 1 #개 이미지 수
          
          if result_dic[key][4] == 1: #예측 값도 개라면
            n_class_cdog += 1 #이미지가 개이고, 예측도 개로 정확히 예측한 경우
            n_match_breed += 1 #이미지가 개면서 품종 예측까지 일치(최종 목적)
            
        else: #이미지가 개가 아니라면 
          if result_dic[key][4] == 0: #예측한 값이 개가 아니라면
            n_class_cnotd += 1 #이미지가 개가 아니면서, 예측도 개가 아닌 걸로 정확히 예측한 경우
            
      else: #이미지 이름과 예측 레이블 값이 일치하지 않는다면
        if result_dic[key][3] == 1: #이미지가 개라면
          n_pet_dog += 1 #개 이미지 수
          
          if result_dic[key][4] == 1: #예측한 값이 개라면
            n_class_cdog += 1 #이미지가 개이고, 예측도 개로 정확히 예측한 경우
            
        else: #이미지가 개라면
          if result_dic[key][4] == 0: #예측한 값이 개가 아니라면
            n_class_cnotd += 1 #이미지가 개가 아니면서, 예측도 개가 아닌 걸로 정확히 예측한 경우
            
    n_pet_notd = n_images - n_pet_dog #개가 아닌 이미지 수 = 총 이미지 수 - 개의 이미지 수
    pct_corr_dog = n_class_cdog / n_pet_dog * 100
    #정확히 예측한 비율(개인 경우만 포함) = 이미지가 개이고, 예측도 개로 정확히 예측한 경우 / 개 이미지 수 * 100.0
    pct_corr_notdog = n_class_cnotd / n_pet_notd * 100
    #정확히 예측한 비율(개가 아닌 경우만) = 이미지가 개가 아니면서, 예측도 개가 아닌 걸로 정확히 예측한 경우 / 개 이미지 수 * 100.0
    pct_corr_breed = n_match_breed / n_pet_dog * 100
    #정확히 예측한 비율(최종 목적) = 이미지가 개면서 품종 예측까지 일치(최종 목적) / 개 이미지 수 * 100.0
    
    print("\n ** Function's Statistics:")
    print("N Images: %2d  N Dog Images: %2d  N NotDog Images: %2d \nPct Corr dog: %5.1f Pct Corr NOTdog: %5.1f Pct Corr Breed: %5.1f"
         % (results_stats_dic["n_images"], results_stats_dic["n_dogs_img"],
           results_stats_dic["n_notdogs_img"], results_stats_dic["pct_correct_dogs"],
           results_stats_dic["pct_correct_notdogs"], results_stats_dic["pct_correct_breed"]))
    print("\n ** Check Statistics:")
    print("N Images: %2d  N Dog Images: %2d  N NotDog Images: %2d \nPct Corr dog: %5.1f Pct Corr NOTdog: %5.1f Pct Corr Breed: %5.1f"
         % (n_images, n_pet_dog, n_pet_notd, pct_corr_dog, pct_corr_notdog, pct_corr_breed))

    # TODO: 7. Define print_results() function to print summary results, 
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch, True, True)

    # TODO: 1. Define end_time to measure total program runtime
    # by collecting end time
#    sleep(75) #sleep(float)을 사용해, 일정 시간 동안 프로그램을 정지 시킬 수 있다.
    end_time = time() #현재 시각

    # TODO: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", tot_time, "in seconds.")
    print("\n** Total Elapsed Runtime:", str( int( (tot_time / 3600) ) ) + ":" +
          str( int(  ( (tot_time % 3600) / 60 )  ) ) + ":" + 
          str( int(  ( (tot_time % 3600) % 60 ) ) ) )
    #tot_time은 종료 시간 - 시작 시간이므로 여기에서 각각 hh:mm:ss를 구한다.
    #time()으로 가져오는 시간은 단위가 1970년 1월 1일 이후의 초 단위 float이다.
    #int()로 형변환 해 소수점을 자른다.
    # hours = int( (tot_time / 3600) )
    # minutes = int( ( (tot_time % 3600) / 60 ) )
    # seconds = int( ( (tot_time % 3600) % 60 ) )


# TODO: 2.-to-7. Define all the function below. Notice that the input 
# paramaters and return values have been left in the function's docstrings. 
# This is to provide guidance for acheiving a solution similar to the 
# instructor provided solution. Feel free to ignore this guidance as long as 
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser() #파라미터를 파싱할 객체 생성
    #커맨드 라인에서 실행될 때 입력된 파라미터를 파싱한다.
    #ArgumentParser()를 실행해 객체를 생성하게 되면, 커맨드 라인에서 실행할 때 
    #--help 또는 -h 옵션으로 실행에 필요한 파라미터와 설명을 확인할 수 있다.
    
    parser.add_argument("--dir", type = str, default = "pet_images/", 
                    help = "path to the folder images") 
    #add_argument()로 각 변수들을 등록한다.
    #커맨드 라인에서 입력될 첫 번째 파라미터는 이미지 폴더의 경로이다. 
    #변수의 이름("--dir"), 타입(str), 기본값("my_folder/"), help 텍스트를 설정한다.
    #여기서 help는 커맨드 라인에서 입력할 때 추력되는 도움말이다.
   
    parser.add_argument("--arch", type = str, default = "vgg", 
                    help = "chosen model")
    #두 번째 파라미터는 CNN에서 사용할 아키텍처 모델이다.
    
    parser.add_argument("--dogfile", type = str, default = "dogname.txt", 
                    help = "text file that has dognames")
    #세 번째 파라미터는 유효한 dog들의 이름을 저장해 놓은 텍스트 파일이다.
    
    return parser.parse_args()
  	#파싱한 결과를 반환한다.
    #parse_args()를 사용해 add_argument()로 추가한 파라미터들을 접근할 수 있도록 파싱한다.
    #add_argument에서 설정한 이름으로 접근할 수 있다. 


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image 
    files. Reads in pet filenames and extracts the pet image labels from the 
    filenames and returns these label as petlabel_dic. This is used to check 
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)  
    """
    
    in_files = listdir(image_dir)
    #os package의 listdir 메서드는 폴더 내의 파일에서 모든 파일 이름을 검색해 리스트로 반환한다.
    #여기서는 pet_images의 각 파일 이름을 가져온다.
    
    petlabels_dic = dict() #레이블 저장할 딕셔너리
    
    for idx in range(0, len(in_files), 1): #in_files의 수 만큼 index로 loop 
      if in_files[idx][0] != ".": #파일명이 "." 으로 시작하지 않는다면 
        #Python은 Swift와 달리 index로 subscript 가능하다.
        image_name = in_files[idx].split("_") #"_"로 나눠서 앞 부분이 이미지 이름
        pet_label = ""
        
        for word in image_name: #image_name에서 한 글자씩 loop
          if word.isalpha(): #알파벳으로만 된 경우
            pet_label += word.lower() + " " #소문자로 저장한다.
            
        pet_label = pet_label.strip() #공백 제거
        
        if in_files[idx] not in petlabels_dic: #딕셔너리에 존재하지 않으면(key)
          petlabels_dic[in_files[idx]] = pet_label #딕셔너리에 key - value로 추가
        else:
          print("Warning: Duplicate files exist in directory",
               in_files[idx])
        #이 프로젝트에선, 파일명이 해당 이미지의 품종(혹은 분류명(개가 아닐 수도 있다))를 나타낸다.
        #따라서 파일이름에서 필요한 string부분만 빼내서 정답 딕셔너리를 만든다.
        #파일명 - 정답 레이블 딕셔너리가 된다.
          
    return petlabels_dic


def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifer labels and 0 = no match between labels
    """
    results_dic = dict() #빈 딕셔너리 생성. 개 품종을 저장함
    
    for key in petlabel_dic: #파일이름 - 개 품종 label로 된 딕셔너리
      model_label = classifier(images_dir + key, model) #classifier에서 정의한 helper Pytorch
      #이미지를 넣어주면 개의 종류 학습된 모델로 예측한 label을 반환
      #여러 품종으로 예측할 수도 있다.
      
      model_label = model_label.lower() #소문자로
      model_label = model_label.strip() #공백 제거
      
      truth = petlabel_dic[key] #target. 정답 레이블 #개가 아닌 것들도 섞여 있다.
      found = model_label.find(truth) #예측한 값과 정답 레이블이 일치하면 found. 아니면 not found.
      #이미지의 레이블은 한 단어로 되어 있으므로, find로 쉽게 비교할 수 있다.
      #find는 일치하는 단어의 첫 index를 반환한다. 찾지 못한 경우는 -1을 반환한다.
     
      if found >= 0: #일치하는 것이 있다면, 예측한 개의 품종이 이미지 레이블에 일부분만 속하는 경우를 걸러내야 한다.
        #ex. (polecat, catamount), (foxhound, English foxhound) ...
        if ( (found == 0 and len(truth) == len(model_label)) or 
            #첫 인덱스부터 일치하고 길이가 같다면 #정답
             ( ( (found == 0) or (model_label[found - 1] == " ") ) and 
              #첫 인덱스부터 일치하거나 model_label의 공백 이후 부터 일치하는 경우 and
               ( (found + len(truth) == len(model_label)) or
                #일치하는 부분이 마지막 부분이거나
                 (model_label[found + len(truth): found + len(truth) + 1] in (",", " "))
                #일치하는 부분 뒤로 공백이나 쉼표 있는 경우 #세 조건 일치해야 정답
               )
             )
           ):
            if key not in results_dic: #정답인 경우
              results_dic[key] = [truth, model_label, 1] #딕셔너리에 추가
              #품종(정답), 예측값, T/F 
        else: #일치하는 부분이 일부분 있는 정답 아닌 경우
          if key not in results_dic:
            results_dic[key] = [truth, model_label, 0] #딕셔너리에 추가
            #품종(정답), 예측값, T/F 
      else: #일치하는 부분이 아예 없는 정답 아닌 경우
        if key not in results_dic: #딕셔너리에 추가
          results_dic[key] = [truth, model_label, 0]
          #품종(정답), 예측값, T/F 
            
    return results_dic      


def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the 
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """           
    dognames_dic = dict() #빈 딕셔너리 생성
    
    with open(dogsfile, "r") as infile:
      line = infile.readline() #한 줄을 읽는다.
      
      while line != "": #EOF(end-of-file)에 이를 때까지 반복
        line = line.rstrip() #오른쪽 공백을 지운다. 
        
        if line not in dognames_dic: #추가 되지 않은 개 품종이라면
          dognames_dic[line] = 1 #추가
        else: #같은 품종이 있다면 오류
          print("** Warning: Duplicate dognames", line)
          
        line = infile.readline() #다음 줄을 읽는다.
        
      #dognames_dic은 개 품종의 딕셔너리가 된다. 품종이름(key) - 1(value)
      #value는 의미가 없는 더미 값이다. 딕셔너리가 속도가 훨씬 빠르기 때문에 최적화를 위해 사용.
      
      for key in results_dic:
        #index 0 : 정답. 실제 분류명(개가 아닌 것도 섞여 있다). 이미지 이름
        #index 1 : CNN 모델이 예측한 레이블(여러 개의 값이 될 수도 있다.) 
        #index 2 : 이미지 이름과 예측 레이블 값의 일치 여부. 0이면 Fasle(예측 맞지 않음), 1이면 True(예측 맞음).
        #index 3 : 정답. 이미지가 개인지 아닌지 여부. 0이면 Fasle(개 아님), 1이면 True(개)
        #index 4 : CNN 모델이 이미지를 개인지 아닌지 예측한 값. 0이면 Fasle(개 아님), 1이면 True(개)
        if results_dic[key][0] in dognames_dic: #이미지 명이 개 품종 딕셔너리에 있다면 (이미지는 개)
          if results_dic[key][1] in dognames_dic: #모델이 예측한 레이블이 개 품종 딕셔너리에 있다면 (예측 값은 개)
            results_dic[key].extend((1, 1)) #index3, 4를 (1, 1)로 확장한다.
            #extend로 value를 확장해 줄수 있다(value가 배열이므로).
          else: #모델이 예측한 레이블이 개 품종 딕셔너리에 없다면 (예측 값은 개가 아님)
            results_dic[key].extend((1, 0)) #index3, 4를 (1, 0)로 확장한다.
        else: #이미지 명이 개 품종 딕셔너리에 없다면 (이미지는 개가 아님)
          if results_dic[key][1] in dognames_dic: #모델이 예측한 레이블이 개 품종 딕셔너리에 있으면 (예측 값은 개)
            results_dic[key].extend((0, 1)) #index3, 4를 (0, 1)로 확장한다.
          else: #모델이 예측한 레이블이 개 품종 딕셔너리에 없으면 (예측 값은 개가 아님)
            results_dic[key].extend((0, 0)) #index3, 4를 (0, 0)로 확장한다.


def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model 
    architecture on classifying images. Then puts the results statistics in a 
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
    """
    results_stats = dict() #빈 딕셔너리 생성
    
    results_stats["n_dogs_img"] = 0 #개 이미지의 수
    results_stats["n_match"] = 0 #예측 결과가 정답과 일치한 수
    results_stats["n_correct_dogs"] = 0 #이미지가 개이고, 예측도 개로 정확히 예측한 경우
    results_stats["n_correct_notdogs"] = 0 #이미지가 개가 아니면서, 예측도 개가 아닌 걸로 정확히 예측한 경우
    results_stats["n_correct_breed"] = 0 #이미지가 개면서 품종 예측까지 일치(최종 목적)
    
    #result_dic은 각 key마다 5개의 value를 가지게 된다. adjust_results4_isadog를 통해 index 3, 4가 추가된다.
    #index 0 : 정답. 실제 분류명(개가 아닌 것도 섞여 있다). 이미지 이름
    #index 1 : CNN 모델이 예측한 레이블(여러 개의 값이 될 수도 있다.) 
    #index 2 : 이미지 이름과 예측 레이블 값의 일치 여부. 0이면 Fasle(예측 맞지 않음), 1이면 True(예측 맞음).
    #index 3 : 정답. 이미지가 개인지 아닌지 여부. 0이면 Fasle(개 아님), 1이면 True(개)
    #index 4 : CNN 모델이 이미지를 개인지 아닌지 예측한 값. 0이면 Fasle(개 아님), 1이면 True(개)
    
    for key in results_dic:
      if results_dic[key][2] == 1: #예측 결과가 정답과 일치한 경우(이미지가 개가 아닌 경우에도 맞을 수 있다.)
        results_stats["n_match"] += 1
      
      if sum(results_dic[key][2:]) == 3: #예측 결과가 정답과 일치하고, 이미지가 개이면서, 이미지 예측 값도 개인 경우
        #index 2, 3, 4의 합이 3. 즉, 셋 다 1이어야 한다.
        results_stats["n_correct_breed"] += 1
        
      if results_dic[key][3] == 1: #이미지가 개인 경우
        results_stats["n_dogs_img"] += 1
        
        if results_dic[key][4] == 1: #이미지를 개로 예측한 경우
          #즉, 이미지가 개이면서, 예측도 개로 정확히 예측한 경우
          results_stats["n_correct_dogs"] += 1

      else: #이미지가 개가 아닌 경우
        if results_dic[key][4] == 0: #이미지를 개로 예측하지 않은 경우
          #즉, 이미지가 개가 아니면서, 예측도 개가 아닌 걸로 정확히 예측한 경우
          results_stats["n_correct_notdogs"] += 1 
    
    results_stats["n_images"] = len(results_dic) #총 이미지 수 = 예측의 총 수
    results_stats["n_notdogs_img"] = results_stats["n_images"] - results_stats["n_dogs_img"]
    #개가 아닌 이미지 수 = 총 이미지 수 - 개의 이미지 수
    results_stats["pct_match"] = results_stats["n_match"] / results_stats["n_images"] * 100.0
    #정확히 예측한 비율(개가 아닌 경우도 포함) = 예측 결과가 정답과 일치한 경우의 수(개가 아닌 경우도 포함) / 총 이미지 수  * 100.0 
    results_stats["pct_correct_dogs"] = results_stats["n_correct_dogs"] / results_stats["n_dogs_img"] * 100.0
    #정확히 예측한 비율(개인 경우만 포함) = 이미지가 개이고, 예측도 개로 정확히 예측한 경우 / 개 이미지 수 * 100.0
    results_stats["pct_correct_breed"] = results_stats["n_correct_breed"] / results_stats["n_dogs_img"] * 100.0
    #정확히 예측한 비율(최종 목적) = 이미지가 개면서 품종 예측까지 일치(최종 목적) / 개 이미지 수 * 100.0
    
    if results_stats["n_notdogs_img"] > 0: #개가 아닌 이미지가 있다면
      results_stats["pct_correct_notdogs"] = results_stats["n_correct_notdogs"] / results_stats["n_notdogs_img"] * 100.0
      #정확히 예측한 비율(개가 아닌 경우만) = 이미지가 개가 아니면서, 예측도 개가 아닌 걸로 정확히 예측한 경우 / 개가 아닌 이미지 수 * 100.0
    else: #개가 아닌 이미지가 없었다면
      results_stats["pct_correct_notdogs"] = 0.0
      
    return results_stats
  

def print_results(results_dic, results_stats, model, print_incorrect_dogs = False, print_incorrect_breed = False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """    
    print("\n\n*** Resilts Summary for CNN Model Architecture", model.upper(), "***") #upper로 대문자로 모델명 표기
    print("%20s: %3d" % ("N Images", results_stats["n_images"])) #이미지 수 
    print("%20s: %3d" % ("N Dog Images", results_stats["n_dogs_img"])) #개 이미지 수
    print("%20s: %3d" % ("N Not-Dog Images", results_stats["n_notdogs_img"])) #개가 아닌 이미지 수
    print(" ")
    
    for key in results_stats:
      if key[0] == "p": #키 값이 p로 시작하는 경우 (백분율인 경우)
        #위에서 딕셔너리에 넣을 때, 백분율은 앞에 pct를 붙였기 때문
        print("%20s: %5.1f" % (key, results_stats[key])) #key와 value 출력
        
    if (print_incorrect_dogs and #print_incorrect_dogs 파라미터를 True로 한 경우 #잘못 분류된 개 이미지를 출력한다.
        #개 이미지가 아닌데, 개로 분류
       	( (results_stats["n_correct_dogs"] + results_stats["n_correct_notdogs"])
         #이미지가 개이고, 예측도 개로 정확히 예측한 경우 수 + #이미지가 개가 아니면서, 예측도 개가 아닌 걸로 정확히 예측한 경우 수
         #즉 이 두 개의 합이 전체 제대로 분류한 경우 수가 된다.
        	!= results_stats["n_images"])
        	#전체 이미지 수와 같지 않을 때 
        	#즉 잘못 분류한 것이 있다는 뜻
       ):
      print("\n INCORRECT Dog/Not Dog Assignments:")
      
      for key in results_dic:
        if sum(results_dic[key][3:]) == 1: #index3, 4의 합이 0(0, 0)이나 2(1, 1)가 나와야 제대로 분류한 것이다.
          #즉 index 3, 4 합이 1이 나왔다는 것은 오류가 있다는 뜻.
          print("Real : %-26s	Classifier: %30s" % (results_dic[key][0], results[key][1]))
          #index 0 : 정답. 실제 분류명(개가 아닌 것도 섞여 있다.)
          #index 1 : CNN 모델이 예측한 레이블(여러 개의 값이 될 수도 있다.) 
          
    if (print_incorrect_breed and #print_incorrect_breed 파라미터를 True로 한 경우 #잘못 분류된 개 품종을 출력한다.
        #개로 분류했지만, 품종이 맞지 않음
       	(results_stats["n_correct_dogs"] != results_stats["n_correct_breed"])
        #이미지가 개이고, 예측도 개로 정확히 예측한 경우 != 이미지가 개면서 품종 예측까지 일치(최종 목적)
        #모델이 개로 분류한 사진의 총 수와 품종 예측까지 정확히 한 수가 일치해야 오류가 없는 것이다.
        #즉 두 수가 같지 않으면 오류가 있다는 뜻.
       ):
      print("\n INCORRECT Dog Breed Assignments:")
      
      for key in results_dic:
        if (sum(results_dic[key][3:]) == 2 and #index3, 4의 합이 2(1, 1) : 이미지가 개이고 모델이 개로 예측
           	results_dic[key][2] == 0 ): #index2가 0이면, 모델의 예측 품종과 실제 품종이 다르다.
          #즉 이미지가 개이고, 모델이 개로 예측 했지만, 품종이 실제와 다른 경우
          print("Real: %-26s	Classifier: %-30s" % (results_dic[key][0], results_dic[key][1]))
          #index 0 : 정답. 실제 분류명(개가 아닌 것도 섞여 있다.)
          #index 1 : CNN 모델이 예측한 레이블(여러 개의 값이 될 수도 있다.) 
                
                
# Call to main function to run the program
if __name__ == "__main__":
    main()
    
#sh run_models_batch.sh를 실행해 모델을 비교해 볼 수 있다.
#서로 다른 세가지 모델을 실행 시키고 결과를 text파일로 볼 수 있다.
