def groupItemsBySingleKeyIndex(listItems, keyIndex):
  dic_itemGroups={}
  for item in listItems:
    key=str(item[keyIndex]) 
    dic_itemGroups.setdefault(key, []).append(item)
   
  return dic_itemGroups