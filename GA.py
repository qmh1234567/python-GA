# 该程序可以求函数的最大值，更换函数则需要进行四处修改
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from matplotlib import animation


# 使用遗传算法找函数的最大值点
population_size=500  #种群数量
generations=200  #迭代次数
# chrom_length=22    # 第二处修改  编码的二进制位数
chrom_length=10 #染色体长度
pc=0.6  # 交配概率
pm=0.01 # 变异概率
genetic_population =[] # 种群的基因编码
population =[] # 种群对应的十进制数值
fitness =[] # 适应度
fitness_mean=[] # 平均适应度
optimum_solution =[] #每次迭代所获最优解

# 为染色体进行0，1编码，生成初始种群
def chrom_encoding():
    for i in range(population_size):
        population_i=[]
        for j in range(chrom_length):
            # 生成每个个体的染色体编码
            population_i.append(random.randint(0,1))
        genetic_population.append(population_i)


# 对染色体进行解码，将二进制转化为十进制
def chrom_decoding():
    population.clear()
    for i in range(population_size):
        value = 0
        for j in range(chrom_length):
            value += genetic_population[i][j]*(2**(chrom_length-1-j))
        # 浮点数表示 在10以内
        population.append(value*10/(2** (chrom_length) - 1))
        # 解码： 区间下限+二进制转化后的十进制数*区间长度（上限-下限）/（2^二进制位数-1）
        # population.append(-1+value*3/(2**(chrom_length)-1))   #第四处修改


# 计算每个染色体的适应度
def calculate_fitness():
    sum=0.0
    fitness.clear()
    for i in range(population_size):
        # 计算每个个体的适应度，即函数的值
        # function_value=population[i]*np.sin(10*np.pi*population[i])+2.0  # 第一处修改
        function_value=10*np.sin(5*population[i])+7*np.cos(4*population[i])
        if function_value > 0.0:
            # 计算种群的适应度
            sum+=function_value
            # 将适应度大于0的个体适应度添加至数组
            fitness.append(function_value)
        else:
            fitness.append(0.0)

    # 返回种群的平均适应度
    return sum/population_size


# 获取最大适应度的个体和对应的编号
def best_value():
    max_fitness=fitness[0]
    max_chrom=0
    for i in range(population_size):
        if fitness[i]>max_fitness:
            max_fitness=fitness[i]
            max_chrom=i
    return population[max_chrom],max_chrom,max_fitness


# 采用轮盘赌算法进行选择过程，重新选择与种群数量相等的新种群
# 用numpy的random.choice函数直接模拟轮盘赌算法
def selection():
    # 种群的适应度数组
    fitness_array = np.array(fitness)
    # 从指定的一维数组中生成随机数 para1:一维数组  para2:数组长度 para3:replace=TRUE 代表可以重复 para4:数组中数据出现的概率
    new_population_id = np.random.choice(np.arange(population_size), (population_size),
                                         replace=True, p=fitness_array/fitness_array.sum())
    new_genetic_population = []
    global genetic_population
    for i in range(population_size):
        new_genetic_population.append(genetic_population[new_population_id[i]])
    genetic_population = new_genetic_population



# 进行交配过程
def crossover():
    for i in range(0,population_size-1,2):
        if random.random()<pc:
            # 随机选择交叉点
            change_point=random.randint(0,chrom_length-1)
            temp1=[]
            temp2=[]
            temp1.extend(genetic_population[i][0:change_point])
            temp1.extend(genetic_population[i+1][change_point:])
            temp2.extend(genetic_population[i+1][0:change_point])
            temp2.extend(genetic_population[i][change_point:])
            genetic_population[i]=temp1
            genetic_population[i+1]=temp2

# 进行基因的变异
def mutation():
    for i in range(population_size):
        if random.random()<pm:
            mutation_point=random.randint(0,chrom_length-1)
            if genetic_population[i][mutation_point]==0:
                genetic_population[i][mutation_point]=1
            else:
                genetic_population[i][mutation_point]=0


if __name__ == '__main__':
    # Draw_Fun()
    chrom_encoding() #编码
    optimum_x_slt=[]  # 最优解对应的x值
    for step in range(generations):
        chrom_decoding()#解码
        fit_mean=calculate_fitness() # 种群的平均适应度
        best_finess_x,best_id,best_finess=best_value() # 适应度最高的个体下标和适应度值
        optimum_solution.append(best_finess) # 每次迭代的最佳适应度
        optimum_x_slt.append(best_finess_x)  # 每次迭代的最佳适应度对应的x值
        fitness_mean.append(fit_mean)  # 每次迭代的种群平均适应度
        selection() #选择
        crossover() # 交叉
        mutation()  #变异
        if step==1 or step==10 or step==40:
            # 绘制最优解在函数图像上的分布
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            # 函数图像
            x = np.linspace(0, 10, 500)
            y = [10 * np.sin(5 * i) + 7 * np.cos(4 * i) for i in x]
            # 第三处修改
            # x = np.linspace(-1, 2, 500)
            # y = [i * np.sin(10 * np.pi * i) + 2.0 for i in x]
            # 两张图画在一起，看一下遗传算法最后收敛到的位置
            ax1.plot(x, y)
            # 设置标题
            ax1.set_title('Scatter Plot ')
            # 设置X轴标签
            plt.xlabel('optimum_x_slt')
            # 设置Y轴标签
            plt.ylabel('optimum_solution')
            # 画散点图
            ax1.scatter(optimum_x_slt, optimum_solution,c='r', marker='o')
            if step==40:
                plt.show()


    # 最优解随迭代次数的变化
    fig1=plt.figure(1)
    print(np.shape(optimum_solution))
    # range取不到最后一位
    plt.plot(range(1,generations+1),optimum_solution)
    plt.xlabel('迭代次数',fontproperties='SimHei')
    plt.ylabel('最优解', fontproperties='SimHei')
    # 平均适应度随迭代次数的变化
    fig2=plt.figure(2)
    plt.plot(range(1,generations+1),fitness_mean)
    plt.xlabel('迭代次数', fontproperties='SimHei')
    plt.ylabel('平均适应度', fontproperties='SimHei')

    plt.show()


