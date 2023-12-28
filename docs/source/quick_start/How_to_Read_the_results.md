# How to Read the results

## The structure of the result txt file



After calculating, you will get the results in the folder: /example/.
Here we use the percentage of true positive contacts among the top 𝑁 predicted contacts to evaluate the results of our methods. In addition, we also evaluate the accuracy of 𝑡𝑜𝑝 𝐿/𝑘 (𝑘 =20) where 𝐿 is the length of one monomer of the homodimer. 

Here is the structure of result file:

```
the top {𝑁} predicted contacts to evaluate the results:

Each line consists of 4 elements:
1、ranking of residue pairs            2、Residue1's number
3、Residue2's number                   4、score of residue pairs

```

Example:

```
#The top20predictions:
Number  Residue1  Residue2  Score
1       11        38        0.9674    
2       139       142       0.9418    
3       142       143       0.9282    
4       135       138       0.9234    
5       131       134       0.9189    
6       11        143       0.9125    
7       55        211       0.8962    
8       54        210       0.8929    
9       7         150       0.8927    
10      7         147       0.8895    
11      138       139       0.8814    
12      55        209       0.8787    
13      7         146       0.8754    
14      12        41        0.8740    
15      55        210       0.8694    
16      42        142       0.8658    
17      40        203       0.8596    
18      14        41        0.8593    
19      43        157       0.8588    
20      134       135       0.8547    

```
