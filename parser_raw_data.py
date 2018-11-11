import re
import sys
def parse_results(input_file, output_init='res'):
    input_f = open(input_file, 'r')
    output_conv = open(output_init + '_conv.txt', 'w')
    output_fc = open(output_init + '_fc.txt', 'w')
    output_pool = open(output_init + '_pool.txt', 'w')
    output_drop = open(output_init + '_drop.txt', 'w')
    output_concat = open(output_init + '_concat.txt', 'w')
    conv_signs = ['conv', 'incept', 'res', 'cccp']
#conv_signs = ['conv', 'incept', 'res', 'mix', 'fc','ip','innerproduct','cccp']
    fc_signs = ['ip','fc','innerproduct']
    pool_signs = ['pool']
    drop_signs = ['drop']
    concat_signs = ['concat']
    for line in input_f:
        if len(line.strip()) == 0: continue
        layer_name = line.split()[0].lower()
        if any(conv in layer_name for conv in conv_signs):
            res = []
            items = line.split(':')
            if len(items) == 8:
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-4:] #Output
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[1]) #Filters
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[2]) #Padding
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[3]) #strides
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[5]) #Inputs
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[6]) #Runtime
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[7]) #Power
                output_conv.write(', '.join(res) + '\n')
        if any(fc in layer_name for fc in fc_signs):
            print layer_name,
        #if any(conv in line.split()[0].lower() for conv in fc_signs):
            res = []
            items = line.split(':')
            if len(items) == 6:
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-2:] #Output
                res += ['1', '1'] # paddings
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[3])[-1:] #Filters
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[4]) #Runtime
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[5]) #Power
                output_fc.write(', '.join(res) + '\n')
            if len(items) == 8:
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-4::3] #Output
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[5])[-3:] #Inputs
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[6]) #Runtime
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[7]) #Power
                output_fc.write(', '.join(res) + '\n')
        if any(pool in layer_name for pool in pool_signs):
            res = []
            items = line.split(':')
            if len(items) == 6:
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-4:]  # Output
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[1])[-2:]  # Kernel
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[2])[-2:]  # Stride
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[3])[-3:-1]  # Input
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[4]) #Runtime
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[5])  # Power
                output_pool.write(', '.join(res) + '\n')
        if any(drop in layer_name for drop in drop_signs):
            res = []
            items = line.split(':')
            if len(items) == 5:
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[1])  # Prob
                tmp = re.findall(r"[-+]?\d*\.\d+|\d+", items[2]) #input
                if len(tmp) == 4:
                    res += tmp
                elif len(tmp) == 2:
                    res += [tmp[0], '1', '1', tmp[1]]
                else:
                    print "Dropout layer parsing error!!"
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[3]) #Runtime
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[4])  # Power
                output_drop.write(', '.join(res) + '\n')
        if any(concat in layer_name for concat in concat_signs):
            res = []
            items = line.split(':')
            if len(items) == 4:
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[2]) #Runtime
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[3])  # Power
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-4:]  # Output
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[1])[3::4]  # input, only last dimension
                output_concat.write(', '.join(res) + '\n')
    input_f.close()
    output_conv.close()
    output_fc.close()
    output_pool.close()
    output_drop.close()
    output_concat.close()

if __name__ == '__main__':
    if len(sys.argv) > 2:
        parse_results(sys.argv[1] #input file
                  , sys.argv[2]) #output_initial
    else:
        parse_results(sys.argv[1])
