from .Imgloader import *


def get_result(request, result):
    global results
    results['input_img'] = results['input_img'] + result[0]
    results['output_img'] = results['output_img'] + result[1]


def Reading(input_img_dir, output_img_dir, file_list: list, input_flag, output_flag, input_type, output_type,
            normalize):
    input_img_data = []
    output_img_data = []
    for file_path in file_list:
        input_file_path = input_img_dir + '/' + file_path[:-4] + input_type
        output_file_path = output_img_dir + '/' + file_path[:-4] + output_type
        input_img_data.append(loading_img_as_numpy(input_file_path, input_flag, normalize=normalize))
        output_img_data.append(loading_img_as_numpy(output_file_path, output_flag,
                                                    normalize=normalize))
    return [input_img_data, output_img_data]


def MultiThreadImgReading(input_img_dir, output_img_dir, workersnumber=10, input_flag=None, output_flag=None,
                          input_type='.png', output_type='.png', normalize=True):
    global results
    results = {
        "input_img": [],
        "output_img": []
    }
    file_name_list = get_dir_filename_list(input_img_dir, type=input_type)
    work_step = len(file_name_list) // workersnumber
    pool = threadpool.ThreadPool(workersnumber)
    arg_list = []
    for i in range(0, len(file_name_list), work_step):
        arg_list.append(([input_img_dir, output_img_dir, file_name_list[i:i + work_step], input_flag, output_flag,
                          input_type, output_type, normalize], None))

    requests = threadpool.makeRequests(Reading, arg_list, get_result)
    [pool.putRequest(req) for req in requests]
    pool.wait()

    return results['input_img'], results['output_img']
