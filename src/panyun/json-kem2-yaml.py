import json
import yaml

def main():
    file_path = "kem_app_json1.txt"
    save_file_path = "output.yaml"
    count =0 
    validApps={'meshserver','nginx-server','ng-oper-ser1','test-oper-sidecar-ser','aspnetapp','win-webserver','auth-service','flow-limit-service','encryption-service','message-service','income-gateway-service','outcome-gateway-service','pd-rebound','demoapp','test-ttt0','nodered-demo'}
    resultcontent=""
    with open(file_path, 'r',encoding="utf-8") as f:
         for line in f:
            count+=1
            #line = eval("'{}'".format(line))
            kemapp = json.loads(line)
            appNote=kemapp['notes']
            print(kemapp['appName']+" "+appNote)
            appName=kemapp['appName']
            yamct=yaml.dump(kemapp,allow_unicode=True,sort_keys=False)
            if(appName in validApps):
                 resultcontent+="~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nInstruction: "
                 resultcontent+="请帮我写一个kem应用的定义文件\n" +"Input: 应用的名字是 "+appName+" "+appNote+" 输出格式是yaml \n"
                 resultcontent+="''''''''''''''''''''''''''''''''\n"
                 resultcontent+="```yaml \n"+yamct+"```\n"
    
    with open(save_file_path, 'w', encoding="utf-8") as f2:
          f2.write(resultcontent)
if __name__=="__main__":
    main()