import subprocess as sp
#psutil was installed as a module, using a virtual environment. Cluster did not allow otherwise.
import psutil as ps
import sys
import time
import threading

#samples network in/out bytes every 5 seconds 
def parse_ifstat(rp_pilot_time_in_min):
    #print("TIMESTAMP", "NETWORKIN_BYTES", "NETWORKOUT_BYTES")
    for i in range(rp_pilot_time_in_min * 12):
        result = sp.run(['ifstat', '-t 5'], stdout=sp.PIPE)
        output = result.stdout.decode('utf-8')
        in_bytes = 0
        out_bytes = 0
        for line in output.split('\n'):
            line = line.strip()
            word_list = line.split()
            #print(word_list)
            
            if len(word_list)>8 and word_list[0] != '0' and not word_list[0].startswith('Interface'):
               bytes = word_list[5]
               if bytes.endswith('K'):
                   in_bytes += int(bytes[:-1])*1024
               elif bytes.endswith('M'):
                   in_bytes += int(bytes[:-1])*1024*1024
               elif bytes.endswith('G'):
                   in_bytes += int(bytes[:-1])*1024*1024*1024
               else:
                   in_bytes += int(bytes)

                
            if len(word_list)>8 and word_list[0] != '0' and not word_list[0].startswith('Interface'):
               bytes = word_list[7]
               if bytes.endswith('K'):
                   out_bytes += int(bytes[:-1])*1024
               elif bytes.endswith('M'):
                   out_bytes += int(bytes[:-1])*1024*1024
               elif bytes.endswith('G'):
                   out_bytes += int(bytes[:-1])*1024*1024*1024
               else:
                   out_bytes += int(bytes)
        
        print("NETWORK:", time.time(), in_bytes, out_bytes, flush=True)
        time.sleep(5)




#Samples CPU utilization every 5 seconds
def printCPUUtil(rp_pilot_time_in_min):
    for i in range(rp_pilot_time_in_min * 12):
        print("CPU:", time.time(), str( ps.cpu_percent(1)) + "%", flush=True)
        time.sleep(4)


#samples disk bytes read/written per second every 5 seconds
def parse_iostat(rp_pilot_time_in_min):
    for i in range(rp_pilot_time_in_min*12):
        
        result = sp.run(['iostat','5', '2'], stdout=sp.PIPE)
        output = result.stdout.decode('utf-8')
        
        read_bytes = 0
        written_bytes = 0
        obsolete_line = False
        line_list = output.split('\n')
        index = 0
        while index < len(line_list):            
            if obsolete_line == False and line_list[index].startswith('avg-cpu'):
                obsolete_line = True
                index += 1
                continue
                
            if obsolete_line == True and line_list[index].startswith('avg-cpu'):
                index += 1
                break
            index += 1
        index += 1    

         
        while index < len(line_list):
            if line_list[index].startswith('Device'):
                index += 1
                break
            index += 1
 
        while index < len(line_list):
            line = line_list[index].strip()
            if line == "":
                break
            word_list = line.split()
            # index 2 and 3 are average kbytes read and written per second in 5 seconds time
            read_bytes += float(word_list[4])*1024
            written_bytes += float(word_list[5])*1024
            index += 1 
        print("DISK:", time.time(), read_bytes, written_bytes)
def parse_free(rp_pilot_time_in_min):

    for i in range(rp_pilot_time_in_min*12):
        result = sp.run(['free','-m'], stdout=sp.PIPE)
        output = result.stdout.decode('utf-8')
        for line in output.split('\n'):
            #print(line, "HELLO")
            if line.startswith("Mem"):
                words = line.strip().split()
                mem_util = int((float(words[1])-float(words[3]))/float(words[1])*100)
                print("MEMORY:", time.time(), str(mem_util)+'%', flush=True)
        time.sleep(5)


def parse_netstat(rp_pilot_time_in_min):
    #print("TIMESTAMP", "NETWORKIN_BYTES", "NETWORKOUT_BYTES")
    for i in range(rp_pilot_time_in_min * 12):
        result = sp.run(['netstat', '-i'], stdout=sp.PIPE)
        output = result.stdout.decode('utf-8')
        in_bytes = 0
        out_bytes = 0
        for line in output.split('\n'):
            line = line.strip()
            word_list = line.split()
            #print(word_list)
            
            if len(word_list) == 11 and not word_list[0].startswith('Iface'):
               in_bytes += int(word_list[2]) * int(word_list[1])
               out_bytes += int(word_list[6]) * int(word_list[1])
                
        time.sleep(5)
        result = sp.run(['netstat', '-i'], stdout=sp.PIPE)
        output = result.stdout.decode('utf-8')
        in_bytes2 = 0
        out_bytes2 = 0
        for line in output.split('\n'):
            line = line.strip()
            word_list = line.split()
            #print(word_list)

            if len(word_list) == 11 and not word_list[0].startswith('Iface'):
               in_bytes2 += int(word_list[2]) * int(word_list[1])
               out_bytes2 += int(word_list[6]) * int(word_list[1])
        #print(in_bytes2 , in_bytes , in_bytes2-in_bytes)
        print("NETWORK:", time.time(), in_bytes2-in_bytes, out_bytes2-out_bytes, flush=True)
        



if __name__ == '__main__':
    runtime_mins = int(sys.argv[1])
    t1 = threading.Thread(target=printCPUUtil, args=(runtime_mins,))
    t2 = threading.Thread(target=parse_netstat, args=(runtime_mins,))
    t3 = threading.Thread(target=parse_iostat, args=(runtime_mins,)) 
    t4 = threading.Thread(target=parse_free, args=(runtime_mins,))
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
