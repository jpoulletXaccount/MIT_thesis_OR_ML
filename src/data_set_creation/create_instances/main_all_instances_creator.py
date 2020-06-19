
import os

from src.data_set_creation.create_instances import instance_creator


if __name__ == '__main__':

    cap = 200
    list_num = [600,1000,1500,2000,3000,5000,7500,10000,15000,20000]
    for num in list_num:
        for i in range(0,10):
            output_dir = "/Users/jpoullet/Documents/MIT/Thesis/benchmarks/cvrptw/created_new/" + str(num)+ "_customers"
            output_file = output_dir + "/test_" + str(i) +".txt"
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            inst_creat = instance_creator.InstanceCreator(output_file=output_file,
                                                          cap=cap,
                                                          number_customer=num)
            inst_creat.generate_instance()

