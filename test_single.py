import pickle
from tqdm import tqdm
import ray
from ray_func import  wait
from utils import downsample_protocol_result
<<<<<<< HEAD
protocol_results = pickle.load(file=open('dataset/down_data.pickle','rb'))
# measured_bit = [0, 1, 22, 24, 45,46, 47, 48, 49,55,57,59,71,80,91,94,99,109]

# protocol_result = downsample_protocol_result(protocol_results,measured_bit)


# # # measured_bit = [0, 1, 3, 5, 6, 8, 11, 13, 18, 22, 25, 26, 27, 28, 29, 30, 34, 35]
# # # protocol_result = downsample_protocol_result(protocol_results,measured_bit)


# with open('read_test/readout_error_mitigation-test/dataset/18_data.pickle', 'wb') as f:
#     pickle.dump(protocol_result, f)


# # protocol_results = pickle.load(file=open('./dataset/protocal_result_18bit.pkl','rb'))
fidelity_single=[]
@ray.remote
def fidelity_single(index,protocol_results):
    f_1=[]
    f_0=[]
    for bitstring1, state_count1 in protocol_results.items():
        str1=[i for i in bitstring1]
        count=0
        if str1[index]=='2':
            continue
        elif str1[index]=='1':
            for bitstring2, state_count2 in state_count1.items():
                str2=[i for i in bitstring2]
                if str2[index]=='1':
                    count=count+state_count2
            f_1.append(count)
        else:
            for bitstring2, state_count2 in state_count1.items():
                str2=[i for i in bitstring2]
                if str2[index]=='0':
                    count=count+state_count2
            f_0.append(count)   
    x_1=sum(f_1)
    x_0=sum(f_0)
    y_1=x_1/len(f_1)
    y_0=x_0/len(f_0)

    return y_0,y_1

protocol_result_token = ray.put(protocol_results)

futures = []

for index in range(131):
    futures.append(fidelity_single.remote(index,protocol_result_token))

result_0=[]
result_1=[]
for y_0,y_1 in tqdm(wait(futures, show_progress=True)): #tqdm(as_completed(futures)):
    result_0.append(y_0)
    result_1.append(y_1)

index_fidelity_0=[]
index_fidelity_1=[]
for i in range(131):
    if result_0[i]<0.02:
        index_fidelity_0.append(i)
    if result_1[i]<0.02:
        index_fidelity_1.append(i)

print(list(set(index_fidelity_0).union(set(index_fidelity_1))))
=======
protocol_results = pickle.load(file=open('./dataset/all_task_id_136_result.pkl','rb'))

measured_bit = [0, 1, 3, 5, 6, 8, 11, 13, 18, 22, 25, 26, 27, 28, 29, 30, 34, 35]
protocol_result = downsample_protocol_result(protocol_results,measured_bit)


with open('new_data_18.pickle', 'wb') as f:
    pickle.dump(protocol_result, f)


# protocol_results = pickle.load(file=open('./dataset/protocal_result_18bit.pkl','rb'))
# fidelity_single=[]
# @ray.remote
# def fidelity_single(index,protocol_results):
#     f_1=[]
#     f_0=[]
#     for bitstring1, state_count1 in protocol_results.items():
#         str1=[i for i in bitstring1]
#         count=0
#         if str1[index]=='2':
#             continue
#         elif str1[index]=='1':
#             for bitstring2, state_count2 in state_count1.items():
#                 str2=[i for i in bitstring2]
#                 if str2[index]=='1':
#                     count=count+state_count2
#             f_1.append(count)
#         else:
#             for bitstring2, state_count2 in state_count1.items():
#                 str2=[i for i in bitstring2]
#                 if str2[index]=='0':
#                     count=count+state_count2
#             f_0.append(count)   
#     x_1=sum(f_1)
#     x_0=sum(f_0)
#     y_1=x_1/len(f_1)
#     y_0=x_0/len(f_0)

#     return y_0,y_1

# protocol_result_token = ray.put(protocol_results)

# futures = []

# for index in range(136):
#     futures.append(fidelity_single.remote(index,protocol_result_token))

# result_0=[]
# result_1=[]
# for y_0,y_1 in tqdm(wait(futures, show_progress=True)): #tqdm(as_completed(futures)):
#     result_0.append(y_0)
#     result_1.append(y_1)

# index_fidelity_0=[]
# index_fidelity_1=[]
# for i in range(136):
#     if result_0[i]>0.95:
#         index_fidelity_0.append(i)
#     if result_1[i]>0.95:
#         index_fidelity_1.append(i)

# print(list(set(index_fidelity_0).union(set(index_fidelity_1))))
>>>>>>> a188bb899043160906c28568ca175c5a1528ced8


    



    
        

