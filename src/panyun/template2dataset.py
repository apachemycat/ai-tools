from mako.template import Template
import json
import copy
import itertools

def cartesian_product(arrays):
    return list(itertools.product(*arrays))

CHILD_NAME='child'    
PROP_ENABLED='enabled'
      
def clone_and_split(json_obj,all_splited):
        need_split=False 
        #print(json_obj)
        count =0 
        rootkey=''
        rotvalue={}
        for fkey, fvalue in json_obj.items():
            rootkey=fkey
            rotvalue=fvalue
            count+=1
        if count > 1:
            raise RuntimeError(' json error ,multi params in one json obj '+str(json_obj))  
        #print(rotvalue)  
        if isinstance(rotvalue, list) :
            for theval in rotvalue:
                new_obj= copy.deepcopy(json_obj)
                new_obj[rootkey]=theval
                all_splited.add(json.dumps(new_obj,ensure_ascii=False))
            return    
        elif  not isinstance(rotvalue, dict):
            all_splited.add(json.dumps(json_obj,ensure_ascii=False))
            return
        
        for key, value in rotvalue.items():
            if key!=CHILD_NAME and isinstance(value, list):
                need_split=True
                for item in value:
                    #new_obj = json_obj.copy()
                    new_obj=copy.deepcopy(json_obj)
                    #new_obj[rootkey]= json_obj[rootkey].copy()
                    #new_obj[rootkey]=json_obj[rootkey].copy()
                    new_obj[rootkey][key] = item
                    clone_and_split(new_obj,all_splited)
        haschildlist =False
        if  need_split==False :
               if CHILD_NAME in rotvalue:
                   #特殊处理child元素，需要2个child，并且分裂成2个独立的1纬度数组，以及保留2个元素的数组
                   childs=rotvalue[CHILD_NAME]
                   if isinstance(childs,list):
                    if len(childs)!=2:
                      raise RuntimeError(' json error ,child must only 2 elements ,cur '+str(len(childs))+' json obj '+str(json_obj))       
                    haschildlist =True 
                    for i in  range(len(childs)):
                           ch=childs[i]
                           for chkey, chvalue in ch.items():
                              if isinstance(chvalue, list):
                                need_split=True 
                                for chitem in chvalue:
                                   new_obj = copy.deepcopy(json_obj)
                                   
                                   new_obj[rootkey][CHILD_NAME][i][chkey] = chitem
                                   #print('split child  '+str(new_obj))
                                   clone_and_split(new_obj,all_splited)   
                    
            
        if need_split ==False: 
            if haschildlist:
                    childs=rotvalue[CHILD_NAME]     
                    for i in  range(len(childs)):
                           ch=childs[i]
                           new_obj = copy.deepcopy(json_obj)
                           new_obj[rootkey][CHILD_NAME]= [ch]
                           #print("split childs to single value "+str(new_obj))
                           all_splited.add(json.dumps(new_obj,ensure_ascii=False))
                           #只放一个child作为样本，减少数据爆炸
                           break
           
            splitedset = all_splited[rootkey] if rootkey in all_splited else []
            splitedset.append(json_obj)
            all_splited.add(json.dumps(json_obj,ensure_ascii=False))
             #@print(all_splited)



    
            
def read_template_params(file_path,singleParams,multiParams):
    print("load json params from file "+file_path)
    with open(file_path, 'r',encoding='utf-8') as file:
        data = json.load(file)
    
    all_json_params = set()
    for item in data:
        clone_and_split(item,all_json_params)
    print("total params count "+str(len(all_json_params)))
    # for item in all_json_params:
    #     print(item)
    #     print(">>>>>>>>")  
    grouped_params={}     
    for item in all_json_params:
             puted=item
             json_obj=json.loads(puted)
             for fkey, fvalue in json_obj.items():
                rootkey=fkey
                rotvalue=fvalue
                break
             if  isinstance(rotvalue, dict) and PROP_ENABLED in rotvalue and rotvalue[PROP_ENABLED]==False:
                 puted= json.dumps({rootkey:{PROP_ENABLED:False}},ensure_ascii=False)
                 #continue
                 
             splitedset = grouped_params[rootkey] if rootkey in grouped_params else set()
             #fvalue才是值，目前暂时把完整JSON放入，包括Key
             splitedset.add(puted)
             #print(json_obj)
             #all_splited[rootkey]=splitedset
             grouped_params[rootkey]=splitedset
             #@print(all_splited)
             
    #print("grouped params count "+str(len(grouped_params)))
    #print("grouped params keys "+str(grouped_params.keys()))
    
    for key,value in grouped_params.items():
        #print(value)
        #print("_____________")
        if len(value) >1:
            multiParams[key]=value
        else:
            singleParams[key]=next(iter(value))
    print("single value  params ,total  "+str(len(singleParams)))
    for key,value in singleParams.items():
     print(key+" "+str(value))
    print("multi params ...")
    multi_key_valut_count_info="multi key and values info : "
    for key,values in multiParams.items():
        multi_key_valut_count_info+=key+":"+str(len(values))+" ,"
        print("key "+key+"   values count "+str(len(values)))
        for valset in values:
            print("   "+str(valset))
    print("muti values  params values  end")
    print("muti values  params values sum \r\n ...."+multi_key_valut_count_info)
def gen_all_template_params(single:dict,multi:dict):
    #  single_template_params={}
    #  single_input_key_values={}
    #  for key,value in single.items():
    #      single_template_params[key]=value
     print("muti params  cartesian_product  begin...")
     #笛卡尔积后的多值参数，
     multiparams_producted= cartesian_product(multi.values())   
     multi=None 
     print("muti params  cartesian_product  result ,counts "+str(len(multiparams_producted)))     
     #加入所有单值参数，组成一个完整的参数集合
     single_values=single.values()
     #print(single_values)
     all_pramas_batch=[]
     min_params_tube=None
     min_parms_tube_len=10000000000
     for mutiparam in multiparams_producted:
         combined=[]
         for item in mutiparam:
             combined.append(item)
         for item in single_values:
             combined.append(item)             
         all_pramas_batch.append(combined)
         paramsstrlen=len(str(combined))
         if paramsstrlen < min_parms_tube_len :
              min_parms_tube_len = paramsstrlen
              min_params_tube=combined
              
     #print(str(min_params_tube))
     #for item in  multiparams_producted:
               
        # theObj=values[0]
        
        # for name,value in theObj.items():
        #    all_template_params[key+'.'+name]=value
     return all_pramas_batch
     

def main():
 templateDir= 'data/template'
 tempateFileName='kem-app-yaml.template'
 tempateParamFile='kem-app-yaml.template.json'
 datatoutFile="output.txt"
 template = Template(filename=templateDir+"/"+tempateFileName,input_encoding="utf-8")
 singleParams={}
 multiParams={}
 read_template_params(templateDir+"/"+tempateParamFile,singleParams,multiParams)
#  print(singleParams)
#  print(">>>>>>>>>")
#  print(multiParams)
#key!=childKey and
 all_pramas_batch=gen_all_template_params(singleParams,multiParams) 
 totals=len(all_pramas_batch)
 print(" total data count  "+str(totals)) 
 curlines=0
 with open(datatoutFile, 'w',encoding="utf-8") as file: 
  for cur_batch  in all_pramas_batch :
     
     
     #print(str(type(cur_batch))+" len "+str(len(cur_batch)))
     cur_batch_all_params=[]
     should_removed_inputkeys=set()
     for jsonObjArry in cur_batch :
      json_obj=json.loads(str(jsonObjArry))
      
      
      #print( json_obj)
      #处理转换过滤参数，用于模板友好操作控制
      cur_bach_json_params={}
      for fkey, fvalue in json_obj.items():
             if isinstance(fvalue, dict):
                 # enabled属性改为 属性本身，False的，则从INPUT中剔除
                 if  PROP_ENABLED in fvalue:
                     cur_bach_json_params[fkey]=fvalue[PROP_ENABLED]
                     if fvalue[PROP_ENABLED]==False:
                         should_removed_inputkeys.add(fkey)
                         continue
                 #None参数的，从Input参数中剔除
                    
                 for propkey,propval in fvalue.items():
                   theKey=fkey+"."+propkey
                   if propkey == PROP_ENABLED :
                         continue
                   cur_bach_json_params[theKey]=propval
                   if propval is None:
                       should_removed_inputkeys.add(theKey) 
             else :
                if fvalue is None:
                   should_removed_inputkeys.add(fkey)   
                cur_bach_json_params[fkey]=fvalue
     
      #print(cur_bach_json_params)   
      cur_batch_all_params.append(cur_bach_json_params)

    #  print("keys")
    #  for key in result:
    #      print(key)
    #      print("........")
     
    


     template_body_params={}
     
     for thejsonparam in cur_batch_all_params:
         for key,value in thejsonparam.items():
             template_body_params[key]=value
    
     #print('should_removed_inputkeys')
     
    #  print(template_body_params)#print(result)
     template_input_params={}
     for key,value in template_body_params.items():
        if  not key in should_removed_inputkeys:
            template_input_params[key]=value
     
     #print("all template input params ")
     #print(template_input_params)
     curlines+=1
     file.write(template.render(params=template_body_params,input=template_input_params))
     if curlines% 100==0:
         print("finished count "+str(curlines) + " progress "+str(curlines/totals))
         #break
     #result = list(itertools.combinations(list(cur_batch_all_params), 2))

   

 #print(params)
 #tn='data/template/test.txt'
 
#  params={'readinessProbe':{'value':True,'type':'httpGet','child':[{'name':'111','type':'aaa'},{'name':'222','type':'bbb'}]},
#          'livenessProbe':{'value':True,'type':'httpGet','child':[{'name':'111','type':'aaa'},{'name':'222','type':'bbb'}]}}
 #params.update({'livenessProbe':False})
 #print(params['readinessProbe'])
 #print(t.render(params=params,input={"key":"xxxxx"}))

if __name__ == "__main__":
    

    main()