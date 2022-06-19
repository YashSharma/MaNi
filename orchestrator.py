from main_notebook import *
import sys

if __name__ == "__main__":
    domain_w = sys.argv[1]
    proto_w = sys.argv[2]    
    source_data = sys.argv[3]
    target_data = sys.argv[4]
    for run_num in range(5):
        run_experiment(domain_w=str(domain_w), 
                       proto_w=str(proto_w), 
                       source_domain=source_data, 
                       target_domain=target_data,
                       infomax_loss_type='concat',
                       projection=str(True))    


