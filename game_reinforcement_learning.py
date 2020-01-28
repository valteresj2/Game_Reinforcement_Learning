#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:02:57 2020

@author: valteresj
"""


##game Reinforcement Learning

import numpy as np
from random import sample
import pandas as pd

n_cols=6
n_rows=6

m=np.zeros((n_rows,n_cols))

mapp=np.array([(i,j) for i in range(n_rows) for j in range(n_cols)])

start_l=[i for i in mapp if i[1]==0]
start=tuple(sample(start_l,1)[0])
finish_l=[i for i in mapp if i[1]==(n_cols-1)]
finish=tuple(sample(finish_l,1)[0])



m_blocks=sample([tuple(i) for i in mapp if tuple(i) != start and tuple(i) != finish or (tuple(i) != (start[0]+1,start[1]) and tuple(i) != (start[0],start[1]+1)) ],int(n_rows*n_cols*0.4))




#action=['frente','cima','baixo']

def sum_inside_array(x,g):
    x= x==g
    return True if sum(x)>1 else False




for index,i in enumerate(mapp):
   action_m=[]
   state_m=[]
   rewards_m=[]
   for j in range(3):
       g=i.copy()
       if j==2:
           g[1]=g[1]+1
           if sum(np.apply_along_axis(sum_inside_array, 1, mapp,g))>0:
               action_m.append('frente')
               state_m.append(str(g))
               rewards_m.append(1)
               
       elif j==1:
           g[0]=g[0]+1
           if sum(np.apply_along_axis(sum_inside_array, 1, mapp,g))>0:
               action_m.append('baixo')
               state_m.append(str(g))
               rewards_m.append(0.5)
       elif j==0 and g[0]>0:
           g[0]=g[0]-1
           if sum(np.apply_along_axis(sum_inside_array, 1, mapp,g))>0:
               action_m.append('cima')
               state_m.append(str(g))
               rewards_m.append(0.5)
#       elif j==0 and g[0]==0:
#           if sum(np.apply_along_axis(sum_inside_array, 1, mapp,g))>0:
#               action_m.append('cima')
#               state_m.append(str(g))
           
   if len(action_m)>0:
       if index==0:
           state=[str(i)]*len(action_m)
           action=action_m
           new_state=state_m
           ru=np.random.uniform(0,1,len(action_m))
           ru=ru/sum(ru)
           prob=list(ru)
           rewards=rewards_m
       else:
           state.extend([str(i)]*len(action_m))
           action.extend(action_m)
           new_state.extend(state_m)
           ru=np.random.uniform(0,1,len(action_m))
           ru=ru/sum(ru)
           prob.extend(list(ru))
           rewards.extend(rewards_m)
          
table_prob=pd.DataFrame({'Current_State':state,'Action':action,'New_state':new_state,'Prob':prob})
table_rew=pd.DataFrame({'Current_State':state,'Action':action,'Rewards':rewards})

policy=[]
state_m=[]
for i in mapp:
    policy.append(np.random.poisson(8,1)[0])
    state_m.append(str(i))
    
table_policy=pd.DataFrame({'State':state_m,'Policy':policy})
table_policy.index=state_m

start=str(np.array(list(start)))
finish=str(np.array(list(finish)))
m_blocks=[str(np.array(list(i))) for i in m_blocks]

gamma=1
update_state=start
iteration=5000
cont=0
cont_gain=0
cont2=0
while True:
    cont2+=1
    index_tab_pro=np.where(np.array(table_prob.loc[:,'Current_State'])==str(update_state))[0]
    index_tab_rew=np.where(np.array(table_rew.loc[:,'Current_State'])==str(update_state))[0]
    index_tab_policy=np.where(np.array(table_policy.loc[:,'State'])==str(update_state))[0]
    actions=np.array(table_prob.loc[index_tab_pro,'Action'])
    V=[]
    for j in range(len(index_tab_pro)):
        rew=table_rew.loc[index_tab_rew[j],'Rewards']
        prob=table_prob.loc[index_tab_pro[j],'Prob']
        pos_state=table_policy.loc[table_prob.loc[index_tab_pro[j],'New_state'],'Policy']
        V.append(rew+prob*pos_state)
    
    table_policy.loc[update_state,'Policy']=np.max(V)
    chosen_action=actions[np.argmax(V)]
    
    update_state=table_prob.loc[index_tab_pro[np.argmax(V)],'New_state']
    
    if update_state==finish:
        cont+=1
        cont_gain+=1
        update_state=start
        print(cont)
#    if update_state in m_blocks or update_state==finish:
#        update_state=start
#        cont+=1
#        print(cont)
    if cont==iteration or cont2==400:
        break
    print(update_state)
        
    
        
    
        



    
    
           

       
    
    
    
    
       
         
 

