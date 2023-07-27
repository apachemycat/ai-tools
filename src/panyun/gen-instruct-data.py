import json
import random
INSTRUCTION_PATTEN='Instruction:'
INPUT_PATTEN='Input:'
def printData(context,answ):
    ind1=context.find(INSTRUCTION_PATTEN)
    ind2=context.find(INPUT_PATTEN)
    inst=context[ind1+INSTRUCTION_PATTEN.__len__():ind2]
    inp=context[ind2+INPUT_PATTEN.__len__():]
    data = {'instruction': inst.strip(),'input':inp.strip(), 'output': answ.strip()}
    # print("  instruction\n"+data['instruction'])
    # print("  input\n"+data['input'])
    # print("___________________ ")
    
    return data
    
def main():

  templateDir= '.'
  file_path = "output.txt"
  save_file_path = "train-instruct.json"
  save_file_path2 = "test-instruct.json"
  gen_test_file=False
  with open(templateDir+"/"+file_path, 'r',encoding="utf-8") as f:
           # Open file for writing

        context = ''
        answer = ''
        results = []
        # Boolean flag to determine whether to append lines to context or answer list
        newItemBegin = False
        answerBegin=False
        # Loop through file, reading each line
        count = 0
        lineNum=1
        for line in f:
            lineNum+=1
            if count % 100 ==0:
                print("finished "+str(count))
            # Check if current line matches starting or ending marker for context or answer data
            if line.strip() == '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~':
                newItemBegin = True
                if answerBegin:
                    count=count+1
      
                    #print(str(count)+"  line "+str(lineNum))
                    data = printData( context,answer)
                    #data = {'instruction': inst,'input':inp, 'output': answer}
                    results.append(data)
                    #print(data)
                    context=''
                    answer=''
                answerBegin = False
                continue
            if line.strip() == "''''''''''''''''''''''''''''''''":
                if newItemBegin :
                   answerBegin=True
                   continue

            # Append each line to appropriate list based on boolean flag
            if answerBegin==False:
                context+=line
            else:
                answer+=line
        #last one
        data = printData(context,answer)      
        results.append(data)
        #print(data)

        with open(templateDir+"/"+save_file_path, 'w', encoding='utf-8') as f2:  
            json.dump(results,f2,ensure_ascii=False)
        if gen_test_file :
                
            with open(templateDir+"/"+save_file_path2, 'w', encoding='utf-8') as f3:  
                sample_size = int(len(results) * 0.05) # Determine the size of the sample based on the desired percentage
                sample = random.sample(results, sample_size) # Use `random.sample()` to select a random subset of the list

                json.dump(sample,f3,ensure_ascii=False)     


    


if __name__=="__main__":
 main()

 