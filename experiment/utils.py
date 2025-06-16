import xml.dom.minidom
import copy

# Model list
model_dict = {
    "meta-llama/Llama-3.1-8B-Instruct": '',  
    "meta-llama/Llama-3.1-70B-Instruct": '', 
    "meta-llama/Llama-3.2-3B-Instruct": '', 
    "meta-llama/Llama-3.3-70B-Instruct": '', 
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": '',
    "allenai/scitulu-7b": '', 
    "allenai/scitulu-70b": '',  
    "mistralai/Mistral-7B-Instruct-v0.2": '', 
    "mistralai/Mistral-7B-Instruct-v0.3": '',  
    "mistralai/Mixtral-8x7B-Instruct-v0.1": '', 
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": '', 
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": '',  
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": '',
    "Qwen/Qwen2.5-14B-Instruct": '',
    "Qwen/Qwen2.5-32B-Instruct": '',
    "Qwen/Qwen2.5-72B-Instruct": '',
}

# Parse topics
def parse_topics(f, names=["param", "crop"]):
    names_return = copy.deepcopy(names)
    topicList = []
    DOMTree = xml.dom.minidom.parse(f)
    collection = DOMTree.documentElement
    topics = collection.getElementsByTagName("topic")
    for topic in topics:
        topicDict = dict()
        for name in names:
            topicDict["id"] = topic.getAttribute("number")
            try:
                #print (name + ' ' + run)
                topicDict[name] = topic.getElementsByTagName(name)[0].childNodes[0].data
            except Exception:
                #names_return.remove(name)
                continue
        topicList.append(topicDict)

    return topicList, names_return
