import re 
import os 
from thefuzz import process
import pandas as pd 
import torch
from modelscope import AutoModelForCausalLM,AutoTokenizer,GenerationConfig

# 为了更好的匹配出ABCD选项，对LLM的回答进行适当的预处理
def process_before_extraction(response,question,choice_dict):
    # 现在大量的计算机是通过诸如以太网这样的局域网连入广域网的，而局域网与广域网的互联是通过____实现的。
    question_split = question.rstrip("。").split("。")[-1].split("_")
    # ['现在大量的计算机是通过诸如以太网这样的局域网连入广域网的，而局域网与广域网的互联是通过', '', '', '', '实现的']
    
    # 将response中'现在大量的计算机是通过诸如以太网这样的局域网连入广域网的，而局域网与广域网的互联是通过'替换成'答案是:'
    if len(question_split[0].strip()) > 4:
        response=response.replace(question_split[0], "答案是")
        
    # 将response中'实现的'替换成''
    if len(question_split[-1].strip()) > 4:
        response = response.replace(question_split[-1], "")
        
    # 将response中的文字形态答案换成选项
    for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
        response=response.replace(val.rstrip("。"), key)
    return response
def extract_choice(response,choice_list):
    # 答案是A | 选项是A | 应该选A选项
    res=re.search(
        r"(?:(?:选|选择|选定)[：:]?\s*|(?:(?:答案|选项)(?![^ABCD]{0,10}?(?:不|非)[^ABCD]{0,10}?(?:是|选|为|：|:|】))[^ABCD]{0,10}?(?:是|选|为|：|:|】))[^ABCD]{0,10}?)(A|B|C|D)(?:选项)?(?:\)|。|\.|，|,|．|、|A|B|C|D|$|：|:|\)|）)",
        response,
    )
    # A选项正确 | A选项符合题意
    if res is None:
        res=re.search(
            r"(A|B|C|D)(?:选?项)?(?![^ABCD]{0,4}?(?:不|非)[^ABCD]{0,4}?(?:正确|对[的，。：]|符合))[^ABCD]{0,4}?(?:正确|对[的，。：]|符合)",
            response,
        )
    # 直接输出 A
    if res is None:
        res=re.search(r"^[\(（]?(A|B|C|D)(?:。|\)|）|\.|，|,|．|：|:|$)",response)
    # 获取第一个出现的字母
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", response)
    # 求response和4个选项中最相似的那一个作为答案
    if res is None:
        return 'ABCD'[choice_list.index(process.extractOne(response,choice_list)[0])]
    return res.group(1)

# 从response中提取A,B,C,D
def extract_answer(response,q):
    response=process_before_extraction(response,q["question"],{choice:q[choice] for choice in 'ABCD'})
    pred=extract_choice(response,[q[choice] for choice in 'ABCD'])
    return pred

# 加载模型
model_name='qwen/Qwen-7B-Chat-Int4'
tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",trust_remote_code=True).eval()
model.generation_config=GenerationConfig.from_pretrained(model_name,trust_remote_code=True)

# 禁用推理多样性
model.generation_config.top_p=0
model.generation_config.repetition_penalty=1.0

# 测试数据清单
TASK_NAME_MAPPING={
    "computer_network": ["Computer Network", "\u8ba1\u7b97\u673a\u7f51\u7edc", "STEM"],
    "operating_system": ["Operating System", "\u64cd\u4f5c\u7cfb\u7edf", "STEM"],
    "computer_architecture": [
        "Computer Architecture",
        "\u8ba1\u7b97\u673a\u7ec4\u6210",
        "STEM",
    ],
    "college_programming": ["College Programming", "\u5927\u5b66\u7f16\u7a0b", "STEM"],
    "college_physics": ["College Physics", "\u5927\u5b66\u7269\u7406", "STEM"],
    "college_chemistry": ["College Chemistry", "\u5927\u5b66\u5316\u5b66", "STEM"],
    "advanced_mathematics": [
        "Advanced Mathematics",
        "\u9ad8\u7b49\u6570\u5b66",
        "STEM",
    ],
    "probability_and_statistics": [
        "Probability and Statistics",
        "\u6982\u7387\u7edf\u8ba1",
        "STEM",
    ],
    "discrete_mathematics": [
        "Discrete Mathematics",
        "\u79bb\u6563\u6570\u5b66",
        "STEM",
    ],
    "electrical_engineer": [
        "Electrical Engineer",
        "\u6ce8\u518c\u7535\u6c14\u5de5\u7a0b\u5e08",
        "STEM",
    ],
    "metrology_engineer": [
        "Metrology Engineer",
        "\u6ce8\u518c\u8ba1\u91cf\u5e08",
        "STEM",
    ],
    "high_school_mathematics": [
        "High School Mathematics",
        "\u9ad8\u4e2d\u6570\u5b66",
        "STEM",
    ],
    "high_school_physics": ["High School Physics", "\u9ad8\u4e2d\u7269\u7406", "STEM"],
    "high_school_chemistry": [
        "High School Chemistry",
        "\u9ad8\u4e2d\u5316\u5b66",
        "STEM",
    ],
    "high_school_biology": ["High School Biology", "\u9ad8\u4e2d\u751f\u7269", "STEM"],
    "middle_school_mathematics": [
        "Middle School Mathematics",
        "\u521d\u4e2d\u6570\u5b66",
        "STEM",
    ],
    "middle_school_biology": [
        "Middle School Biology",
        "\u521d\u4e2d\u751f\u7269",
        "STEM",
    ],
    "middle_school_physics": [
        "Middle School Physics",
        "\u521d\u4e2d\u7269\u7406",
        "STEM",
    ],
    "middle_school_chemistry": [
        "Middle School Chemistry",
        "\u521d\u4e2d\u5316\u5b66",
        "STEM",
    ],
    "veterinary_medicine": ["Veterinary Medicine", "\u517d\u533b\u5b66", "STEM"],
    "college_economics": [
        "College Economics",
        "\u5927\u5b66\u7ecf\u6d4e\u5b66",
        "Social Science",
    ],
    "business_administration": [
        "Business Administration",
        "\u5de5\u5546\u7ba1\u7406",
        "Social Science",
    ],
    "marxism": [
        "Marxism",
        "\u9a6c\u514b\u601d\u4e3b\u4e49\u57fa\u672c\u539f\u7406",
        "Social Science",
    ],
    "mao_zedong_thought": [
        "Mao Zedong Thought",
        "\u6bdb\u6cfd\u4e1c\u601d\u60f3\u548c\u4e2d\u56fd\u7279\u8272\u793e\u4f1a\u4e3b\u4e49\u7406\u8bba\u4f53\u7cfb\u6982\u8bba",
        "Social Science",
    ],
    "education_science": ["Education Science", "\u6559\u80b2\u5b66", "Social Science"],
    "teacher_qualification": [
        "Teacher Qualification",
        "\u6559\u5e08\u8d44\u683c",
        "Social Science",
    ],
    "high_school_politics": [
        "High School Politics",
        "\u9ad8\u4e2d\u653f\u6cbb",
        "Social Science",
    ],
    "high_school_geography": [
        "High School Geography",
        "\u9ad8\u4e2d\u5730\u7406",
        "Social Science",
    ],
    "middle_school_politics": [
        "Middle School Politics",
        "\u521d\u4e2d\u653f\u6cbb",
        "Social Science",
    ],
    "middle_school_geography": [
        "Middle School Geography",
        "\u521d\u4e2d\u5730\u7406",
        "Social Science",
    ],
    "modern_chinese_history": [
        "Modern Chinese History",
        "\u8fd1\u4ee3\u53f2\u7eb2\u8981",
        "Humanities",
    ],
    "ideological_and_moral_cultivation": [
        "Ideological and Moral Cultivation",
        "\u601d\u60f3\u9053\u5fb7\u4fee\u517b\u4e0e\u6cd5\u5f8b\u57fa\u7840",
        "Humanities",
    ],
    "logic": ["Logic", "\u903b\u8f91\u5b66", "Humanities"],
    "law": ["Law", "\u6cd5\u5b66", "Humanities"],
    "chinese_language_and_literature": [
        "Chinese Language and Literature",
        "\u4e2d\u56fd\u8bed\u8a00\u6587\u5b66",
        "Humanities",
    ],
    "art_studies": ["Art Studies", "\u827a\u672f\u5b66", "Humanities"],
    "professional_tour_guide": [
        "Professional Tour Guide",
        "\u5bfc\u6e38\u8d44\u683c",
        "Humanities",
    ],
    "legal_professional": [
        "Legal Professional",
        "\u6cd5\u5f8b\u804c\u4e1a\u8d44\u683c",
        "Humanities",
    ],
    "high_school_chinese": [
        "High School Chinese",
        "\u9ad8\u4e2d\u8bed\u6587",
        "Humanities",
    ],
    "high_school_history": [
        "High School History",
        "\u9ad8\u4e2d\u5386\u53f2",
        "Humanities",
    ],
    "middle_school_history": [
        "Middle School History",
        "\u521d\u4e2d\u5386\u53f2",
        "Humanities",
    ],
    "civil_servant": ["Civil Servant", "\u516c\u52a1\u5458", "Other"],
    "sports_science": ["Sports Science", "\u4f53\u80b2\u5b66", "Other"],
    "plant_protection": ["Plant Protection", "\u690d\u7269\u4fdd\u62a4", "Other"],
    "basic_medicine": ["Basic Medicine", "\u57fa\u7840\u533b\u5b66", "Other"],
    "clinical_medicine": ["Clinical Medicine", "\u4e34\u5e8a\u533b\u5b66", "Other"],
    "urban_and_rural_planner": [
        "Urban and Rural Planner",
        "\u6ce8\u518c\u57ce\u4e61\u89c4\u5212\u5e08",
        "Other",
    ],
    "accountant": ["Accountant", "\u6ce8\u518c\u4f1a\u8ba1\u5e08", "Other"],
    "fire_engineer": [
        "Fire Engineer",
        "\u6ce8\u518c\u6d88\u9632\u5de5\u7a0b\u5e08",
        "Other",
    ],
    "environmental_impact_assessment_engineer": [
        "Environmental Impact Assessment Engineer",
        "\u73af\u5883\u5f71\u54cd\u8bc4\u4ef7\u5de5\u7a0b\u5e08",
        "Other",
    ],
    "tax_accountant": ["Tax Accountant", "\u7a0e\u52a1\u5e08", "Other"],
    "physician": ["Physician", "\u533b\u5e08\u8d44\u683c", "Other"],
}

# 遍历每个题库
task_result=[]
for task_name,task_info in TASK_NAME_MAPPING.items():
    print(f'---{task_name} begin---')
    
    # id,question,A,B,C,D,answer
    task_csv=pd.read_csv('data/ceval/val/%s_val.csv'%(task_name))
    
    # 回答每一道题
    raw_response=[]
    answers=[]
    for _,q in task_csv.iterrows():
        prompt=f'{q["question"]}\n\nA. {q["A"]}\nB. {q["B"]}\nC. {q["C"]}\nD. {q["D"]}\n'   # 问题
        with torch.no_grad():
            response,_=model.chat(tokenizer,prompt,history=None)    # 调用LLM
            raw_response.append(response)
        ans=extract_answer(response,q)   # 提取答案
        answers.append(ans)
        # print('%s,正确答案:%s,模型答案:%s'%(q['question'],q['answer'],ans))
    
    # 判题
    correctness=[task_csv.iloc[q_no]['answer']==ans for q_no,ans in enumerate(answers)]
    
    # 保存结果
    task_csv['model_response']=raw_response
    task_csv['model_answer']=answers
    task_csv['correctness']=correctness
    os.makedirs('result',exist_ok=True)
    task_csv.to_csv(f'result/{task_name}.csv',index=False)

    # 正确率
    correct_rate=sum(correctness)/len(answers)
    
    # 任务结果清单
    task_result.append({'task_name':task_name,'task_intro':task_info[1],'task_type':task_info[2],'question_num':len(answers),'correct_rate':correct_rate})
    
    print('%s 正确率: %.2f'%(task_name,correct_rate))
    print(f'---{task_name} end---')

# 保存所有任务结果
task_final_result=pd.DataFrame(data=task_result)
task_final_result.to_csv('task_final_result.csv')