import numpy as np
from metrics.predictive_metrics import predictive_score_metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from data_loading import real_data_loading

ori_data = real_data_loading("./data",'sine',24)#['stock', 'knn_air','air','etth1','sine']

def test(ori_data):
    print('Start Testing')
    gen_data = ori_data
    print('Finish Synthetic Data Generation')

    # Performance metrics
    metric_results = dict()
    # 2. Predictive score
    predictive_score = list()
    print('Start predictive_score_metrics')
    for i in range(10):
        print('predictive_score iteration: ', i)
        temp_predict = predictive_score_metrics(ori_data, gen_data)
        predictive_score.append(temp_predict)
    metric_results['predictive'] = np.mean(predictive_score)
    print('Finish predictive_score_metrics compute')
    print(metric_results['predictive'])

test(ori_data)