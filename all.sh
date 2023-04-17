# 1 - train aruco - test aruco
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data -f ./results_vanilla/1/train_aruco/test_aruco -w phi_0_linemod_best_ADD.h5 -o 1 -c false
# 1 -train aruco - test NO aruco
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data -f results_vanilla/1/train_aruco/test_noaruco -w phi_0_linemod_best_ADD.h5 -o 1 -c false --noaruco
# 1 -train NO aruco test aruco
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data -f results_vanilla/1/train_noaruco/test_aruco -w phi_0_noaruco-linemod_best_ADD.h5 -o 1 -c false
# 1 -train No aruco test No aruco
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data -f results_vanilla/1/train_noaruco/test_noaruco -w phi_0_noaruco-linemod_best_ADD.h5 -o 1 -c false --noaruco

# 5 - train aruco - test aruco
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data -f results_vanilla/5/train_aruco/test_aruco -w phi_0_linemod_best_ADD.h5 -o 5 -c false
# 5 -train aruco - test NO aruco
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data -f results_vanilla/5/train_aruco/test_noaruco -w phi_0_linemod_best_ADD.h5 -o 5 -c false --noaruco
# 5 -train NO aruco test aruco
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data -f results_vanilla/5/train_noaruco/test_aruco -w phi_0_noaruco-linemod_best_ADD.h5 -o 5 -c false
# 5 -train No aruco test No aruco
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data -f results_vanilla/5/train_noaruco/test_noaruco -w phi_0_noaruco-linemod_best_ADD.h5 -o 5 -c false --noaruco

# 11 - train aruco - test aruco
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data -f ./results_vanilla/11/train_aruco/test_aruco -w phi_0_linemod_best_ADD-S.h5 -o 11 -c false
# 11 -train aruco - test NO aruco
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data -f results_vanilla/11/train_aruco/test_noaruco -w phi_0_linemod_best_ADD-S.h5 -o 11 -c false --noaruco
# 11 -train NO aruco test aruco
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data -f results_vanilla/11/train_noaruco/test_aruco -w phi_0_noaruco-linemod_object11.h5 -o 11 -c false
# 11 -train No aruco test No aruco
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data -f results_vanilla/11/train_noaruco/test_noaruco -w phi_0_noaruco-linemod_object11.h5 -o 11 -c false --noaruco
