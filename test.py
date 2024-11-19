from docplex.mp.model import Model
from contextlib import redirect_stdout
import os
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 定义问题参数
T = range(1, 100001)  # 时间点集合
N = 7  # 车辆任务
Vehicles = ['1', '2', '3', '4', '5', '6']  # 车辆集合 数量多一点
lines = ['l_1', 'l_2', 'l_3']  # 产线集合
Tl = {'l_1': 10, 'l_2': 10, 'l_3': 10}  # 产线的生产速度/s
H = 10  # 产线任务
G = 10  # 运输时间
r = 400  # 初凝时间
Pv = 10  # 浇筑速度
max_departure_time = 170000  # 最大的出发时间
max_time = 110000  # 最大生产结束时间，大一点
offset = 30
M1 = 1000000  # ∞
C = 1


# 指定只读取前 N 行数据，例如 50 行
N = 50

# 使用绝对路径
file_path = r'data.xlsx'
sheet_name = 'Sheet1'  # 替换为你想要读取的 sheet 名称或索引

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"文件未找到: {file_path}")
else:
    # 读取 Excel 文件
    try:
        # df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl', nrows=N)

        # 初始化空的任务组字典
        task_groups = {}

        # 初始化字典 E、L_T_k_m_2 和 Q
        E = {}
        L_T_k_m_2 = {}
        Q = {}

        # 遍历 DataFrame 来填充 task_groups 字典和 E、L_T_k_m_2 和 Q 字典
        for index, row in df.iterrows():
            task_group = row['任务单号']  # 根据实际列名调整
            subtask = row['运输单号']  # 根据实际列名调整
            start_time = row['任务单开始浇筑时间1']  # 根据实际列名调整
            end_time = row['任务单结束浇筑时间1']  # 根据实际列名调整
            demand = row['子任务需求']  # 根据实际列名调整

            # 更新 task_groups 字典
            if task_group not in task_groups:
                task_groups[task_group] = []

            task_groups[task_group].append(subtask)

            # 更新 E、L_T_k_m_2 和 Q 字典
            E[subtask] = start_time
            L_T_k_m_2[subtask] = end_time
            Q[subtask] = demand

        # 输出 task_groups 字典
        print("任务组字典:")
        print(task_groups)

        # 输出 E、L_T_k_m_2 和 Q 字典
        print("\n最早开始时间字典 (E):")
        print(E)
        print("\n最晚开始时间字典 (L_T_k_m_2):")
        print(L_T_k_m_2)
        print("\n需求字典 (Q):")
        print(Q)

        # 提取所有任务
        tasks = [task for group in task_groups.values() for task in group]

        # 创建模型
        mdl = Model(name="Task Scheduling")
        # mdl.parameters.mip.tolerances.mipgap = 0.05  # 设置 MIP 间隙
        # mdl.parameters.timelimit = 300  # 设置时间限制（秒） 减少收敛时间算

        mdl.parameters.mip.tolerances.mipgap = 0.1  # 调高间隙为 10%
        mdl.parameters.timelimit = 100
        mdl.log_output = True  # 启用日志输出

        # 定义决策变量
        delivery_by_vehicle = {(x, k, n): mdl.binary_var(name=f"delivery_by_vehicle_{x}_{k}_{n}") for x in tasks for k in Vehicles for n in range(N)}  # 分配车辆
        Ts_x = {x: mdl.integer_var(lb=0, ub=max_time, name=f"Ts_x_{x}") for x in tasks}
        production_by_line = {(x, l, h): mdl.binary_var(name=f"production_by_line_{x}_{l}_{h}") for x in tasks for l in lines for h in range(H)}  # 匹配产线
        # 定义其他参数
        # 车
        Tk0_aux2 = {(x, k, l, n, h): mdl.continuous_var(name=f"Tk0_aux2_{x}_{k}_{l}_{n}_{h}") for k in Vehicles for x in
                    tasks for l in lines for n in range(N) for h in range(H)}

        # Tk0_aux2 和相关约束
        for x in tasks:
            for k in Vehicles:
                for l in lines:
                    for n in range(N):
                        for h in range(H):
                            mdl.add_constraint(Tk0_aux2[(x, k, l, n, h)] <= production_by_line[(x, l, h)])
                            mdl.add_constraint(Tk0_aux2[(x, k, l, n, h)] <= delivery_by_vehicle[(x, k, n)])
                            mdl.add_constraint(
                                Tk0_aux2[(x, k, l, n, h)] >= production_by_line[(x, l, h)] + delivery_by_vehicle[
                                    (x, k, n)] - 2)
        Tk0 = {(x, k, n): mdl.continuous_var(name=f"Tk0_{x}_{k}_{n}") for x in tasks for k in Vehicles for n in
               range(N)}
        Tk1 = {(x, k, n): mdl.continuous_var(name=f"Tk1_{x}_{k}_{n}") for x in tasks for k in Vehicles for n in
               range(N)}
        Tk2 = {(x, k, n): mdl.continuous_var(name=f"Tk2_{x}_{k}_{n}") for x in tasks for k in Vehicles for n in
               range(N)}
        Tk3 = {(x, k, n): mdl.continuous_var(name=f"Tk3_{x}_{k}_{n}") for x in tasks for k in Vehicles for n in
               range(N)}
        Tk4 = {(x, k, n): mdl.continuous_var(name=f"Tk4_{x}_{k}_{n}") for x in tasks for k in Vehicles for n in
               range(N)}
        Tk5 = {(x, k, n): mdl.continuous_var(name=f"Tk5_{x}_{k}_{n}") for x in tasks for k in Vehicles for n in
               range(N)}
        Wp = {(x, k, n): mdl.continuous_var(name=f"Wp_{x}_{k}_{n}") for x in tasks for k in Vehicles for n in range(N)}
        # 产线

        Ts_l = {(x, l, h): mdl.continuous_var(name=f"Ts_l_{x}_{l}_{h}") for l in lines for x in tasks for h in
                range(H)}  # 产线l的任务h的开始时间
        Te_l = {(x, l, h): mdl.continuous_var(name=f"Te_l_{x}_{l}_{h}") for l in lines for x in tasks for h in
                range(H)}  # 产线l的任务h的结束时间
        Ts_l_aux = {(x, l, h): mdl.continuous_var(name=f"Ts_l_aux_{x}_{l}_{h}") for l in lines for x in tasks for h in
                    range(H)}
        # 任务
        Tc_x = {(x, l, h): Ts_x[x] + (Q[x] / Tl[l]) * production_by_line[(x, l, h)] for l in lines for x in tasks for h
                in range(H)}  # 生产结束时间
        Wd_1_first = {(x, k, n): mdl.continuous_var(name=f"Wd_1_first_{x}_{k}_{n}") for x in tasks for k in Vehicles for
                      n in range(N)}  # 浇筑等待
        wd2 = {(x, k, n): mdl.continuous_var(name=f"wd2_{x}_{k}_{n}") for x in tasks for k in Vehicles for n in
               range(N)}  # 浇筑等待
        # 工地
        Ts_g = {(x): mdl.continuous_var(name=f"Ts_g_{x}") for x in tasks}  # 工地浇筑开始时间
        Te_g = {(x): mdl.continuous_var(name=f"Te_g_{x}") for x in tasks}  # 工地浇筑结束时间
        Ts_g_aux = {(x): mdl.continuous_var(name=f"Ts_g_aux_{x}") for x in tasks}  # 工地浇筑开始时间
        Te_g_aux = {(x): mdl.continuous_var(name=f"Te_g_aux_{x}") for x in tasks}  # 工地浇筑结束时间
        t = {(x): mdl.continuous_var(name=f"t_{x}") for x in tasks}  # 松弛时间△t

        # 约束
        # 车
        # a车辆的a任务只发车一次
        for k in Vehicles:
            for n in range(N):
                mdl.add_constraint(mdl.sum(delivery_by_vehicle[(x, k, n)] for x in tasks) <= 1)

        # 添加 Tk1 和 Tk0 之间的关系,直接赋值Tk1==生产结束时间(直接赋值跑不出来，增加生产等待时间【min】)
        for x in tasks:
            for k in Vehicles:
                for n in range(N):
                    for h in range(H):
                        for l in lines:
                            production_time = sum(
                                (Q[x] / Tl[l]) * Tk0_aux2[(x, k, l, n, h)] for l in lines for h in range(H))
                            mdl.add_constraint(Tk1[(x, k, n)] >= Tk0[(x, k, n)] + production_time)

        # Tc_x[(x, l, h)]的计算=Ts+production_time
        for x in tasks:
            for l in lines:
                for h in range(H):
                    mdl.add_constraint(Tc_x[(x, l, h)] == Ts_x[x] + (Q[x] / Tl[l]) * production_by_line[(x, l, h)])



        # 生产等待时间
        for k in Vehicles:
            for n in range(N):
                for x in tasks:
                    for l in lines:
                        for h in range(H):
                            mdl.add_constraint(Wp[(x, k, n)] >= Tk1[(x, k, n)] - Tc_x[(x, l, h)])

        # k车的n+1的Tk1>=n的Tk1
        M = 100
        for k in Vehicles:
            for x1 in tasks:
                for x2 in tasks:
                    if x1 != x2:
                        for n in range(N):
                            for n_prime in range(n + 1, N):
                                mdl.add_constraint(Tk1[(x1, k, n)] - Tk1[(x2, k, n_prime)] <= M * (
                                            1 - delivery_by_vehicle[(x1, k, n)]) + M * (
                                                               1 - delivery_by_vehicle[(x2, k, n_prime)]))
        # 计算Tk2
        for x in tasks:
            for k in Vehicles:
                for n in range(N):
                    mdl.add_constraint(Tk2[(x, k, n)] == Tk1[(x, k, n)] + G * delivery_by_vehicle[(x, k, n)])

        UT_x = {x: Q[x] / Pv for x in tasks}  # 浇筑时间
        # 第一个子任务的浇筑等待
        Wd1_aux_first = {(x, k, n): E[x] - Tk2[(x, k, n)] for x in [group[0] for group in task_groups.values()] for k in
                         Vehicles for n in range(N)}
        for x in [group[0] for group in task_groups.values()]:
            for k in Vehicles:
                for n in range(N):
                    mdl.add_constraint(Wd_1_first[(x, k, n)] >= Wd1_aux_first[(x, k, n)])
                    mdl.add_constraint(Wd_1_first[(x, k, n)] >= 0)
        # 计算第一个子任务的 Tk3 已经被定义为 Tk2 加上 Wd_1_first
        for x in [group[0] for group in task_groups.values()]:
            for k in Vehicles:
                for n in range(N):
                    mdl.add_constraint(Tk3[(x, k, n)] == Tk2[(x, k, n)] + Wd_1_first[(x, k, n)])

        # 对于后面任务组，连续浇筑，
        for group_tasks in task_groups.values():
            for i in range(1, len(group_tasks)):
                task_i = group_tasks[i - 1]  # 前一个子任务
                task_i_plus_1 = group_tasks[i]  # 当前子任务
                mdl.add_constraint(Ts_g[task_i_plus_1] <= Te_g[task_i])
                mdl.add_constraint(Te_g[task_i] <= Ts_g[task_i_plus_1])
        # 用任务开始浇筑表示Tk3用任务结束浇筑表示Tk4
        # 确保 Ts_g[x] 等于任务 x 被分配到的车辆 k 和任务编号 n 的浇筑开始时间 Tk3[(x, k, n)]
        for x in tasks:
            for k in Vehicles:
                for n in range(N):
                    # 使用大 M 方法来处理条件约束
                    mdl.add_constraint(Ts_g[x] <= Tk3[(x, k, n)] + M * (1 - delivery_by_vehicle[(x, k, n)]))
                    mdl.add_constraint(Ts_g[x] >= Tk3[(x, k, n)] - M * (1 - delivery_by_vehicle[(x, k, n)]))
                    mdl.add_constraint(Te_g[x] <= Tk4[(x, k, n)] + M * (1 - delivery_by_vehicle[(x, k, n)]))
                    mdl.add_constraint(Te_g[x] >= Tk4[(x, k, n)] - M * (1 - delivery_by_vehicle[(x, k, n)]))

        # 等待时间约束
        for group_tasks in task_groups.values():
            for i in range(1, len(group_tasks)):
                for x in [group_tasks[i]]:
                    for k in Vehicles:
                        for n in range(N):
                            # 计算等待时间
                            mdl.add_constraint(wd2[(x, k, n)] >= Tk3[(x, k, n)] - Tk2[(x, k, n)])
                            # 确保等待时间非负
                            mdl.add_constraint(wd2[(x, k, n)] >= 0)
                            mdl.add_constraint(Tk3[(x, k, n)] >= Tk2[(x, k, n)] + wd2[(x, k, n)])
        # 任务顺序发车，确保x1 x2不分配给同一位置
        for group_tasks in task_groups.values():
            for i in range(len(group_tasks) - 1):
                task_i = group_tasks[i]  # 当前子任务
                task_i_plus_1 = group_tasks[i + 1]  # 下一个子任务
                for k in Vehicles:
                    for n in range(N):
                        for k_prime in Vehicles:
                            for m in range(N):
                                mdl.add_constraint(Tk1[(task_i, k, n)] - Tk1[(task_i_plus_1, k_prime, m)] + C <= M * (
                                            1 - delivery_by_vehicle[(task_i, k, n)]) + M * (
                                                               1 - delivery_by_vehicle[(task_i_plus_1, k_prime, m)])
                                                   )

        # Tk4 的计算
        for x in tasks:
            for k in Vehicles:
                for n in range(N):
                    mdl.add_constraint(Tk4[(x, k, n)] == Tk3[(x, k, n)] + UT_x[x] * delivery_by_vehicle[(x, k, n)])

        # Tk5 的计算
        Tk5 = {(x, k, n): Tk4[(x, k, n)] + G * delivery_by_vehicle[(x, k, n)] for x in tasks for k in Vehicles for n in range(N)}
        # 添加返厂时间约束
        for k in Vehicles:
            for n in range(N - 1):
                for x1 in tasks:
                    for x2 in tasks:
                        if x1 != x2:
                            mdl.add_constraint(Tk5[(x1, k, n)] == Tk0[(x2, k, n + 1)])
        # 确保如果车辆k在n+1有交付任务，则在n也必须有一个交付任务
        for k in Vehicles:
            for n in range(N - 1):
                mdl.add_constraint(
                    mdl.sum(delivery_by_vehicle[(x, k, n + 1)] for x in tasks) <=
                    mdl.sum(delivery_by_vehicle[(x, k, n)] for x in tasks)
                )
        # 产线
        # a产线的a任务只工作一次
        for l in lines:
            for h in range(H):
                mdl.add_constraint(mdl.sum(production_by_line[(x, l, h)] for x in tasks) <= 1)

        # 循环，早于后面的所有任务h+1.....H

        for l in lines:
            for h in range(H - 1):  # 遍历除了最后一个以外的所有时间段
                    for x1 in tasks:
                        for x2 in tasks:
                            if x1 != x2:
                                mdl.add_constraint(Te_l[(x1, l, h)] - Ts_l[(x2, l, h + 1)] <= M * (
                                            1 - production_by_line[x1, l, h]) + M * (
                                                               1 - production_by_line[x2, l, h + 1]))


        for l in lines:
            for h in range(H - 1):
                mdl.add_constraint(
                    mdl.sum(production_by_line[x, l, h + 1] for x in tasks) <=
                    mdl.sum(production_by_line[(x, l, h)] for x in tasks)
                )
        # 任务
        # x+1任务的开始生产时间晚于x
        for group_tasks in task_groups.values():
            for i in range(1, len(group_tasks)):
                task_i = group_tasks[i - 1]  # 前一个子任务
                task_i_plus_1 = group_tasks[i]  # 当前子任务
                mdl.add_constraint(Ts_x[task_i_plus_1] >= Ts_x[task_i])
        # 每个任务分配给产线/车辆一次
        for x in tasks:
            mdl.add_constraint(mdl.sum(delivery_by_vehicle[(x, k, n)] for k in Vehicles for n in range(N)) == 1)
            mdl.add_constraint(mdl.sum(production_by_line[(x, l, h)] for l in lines for h in range(H)) == 1)
        # 初凝时间约束
        for x in tasks:
            for k in Vehicles:
                for n in range(N):
                    for l in lines:
                        for h in range(H):
                            mdl.add_constraint(r >= Tk3[(x, k, n)] - Tc_x[(x, l, h)])
        # 定义产线的完成时间等于发车时间
        for x in tasks:
            for k in Vehicles:
                for l in lines:
                    for n in range(N):
                        for h in range(H):
                            mdl.add_constraint(Tc_x[(x, l, h)] <= Tk1[(x, k, n)])  # 任务生产完成<=车辆发车
                            mdl.add_constraint(Te_l[(x, l, h)] <= Tk1[(x, k, n)])  # 产线生产完成<=车辆发车
        # 定义Ts_l等于产品的生产开始时间
        for x in tasks:
            for l in lines:
                for h in range(H):
                    mdl.add_constraint(Ts_l_aux[(x, l, h)] <= Ts_x[x])
                    mdl.add_constraint(Ts_l_aux[(x, l, h)] <= production_by_line[(x, l, h)] * M1)
                    mdl.add_constraint(Ts_l_aux[(x, l, h)] >= Ts_x[x] + production_by_line[(x, l, h)] - M1)
                    mdl.add_constraint(Ts_l[(x, l, h)] == Ts_l_aux[(x, l, h)])
                    mdl.add_constraint(Te_l[(x, l, h)] == Tc_x[(x, l, h)])  # 产线生产结束时间==产品生产结束时间
        # 创建 late_first 变量字典
        late_first = {}
        for x in tasks:
            for k in Vehicles:
                for n in range(N):
                    late_first[(x, k, n)] = mdl.continuous_var(name=f'late_first_{x}_{k}_{n}')

        # 添加约束
        for x in tasks:
            for k in Vehicles:
                for n in range(N):
                    # 确保 late_first[(x, k, n)] >= Tk3[(x, k, n)] - L_T_k_m_2[x]
                    mdl.add_constraint(late_first[(x, k, n)] >= Tk3[(x, k, n)] - L_T_k_m_2[x])
                    # 确保 late_first[(x, k, n)] >= 0
                    mdl.add_constraint(late_first[(x, k, n)] >= 0)

        # 目标函数：最小化所有任务的完成时间之和
        mdl.minimize(mdl.sum(
            Wd_1_first[(x, k, n)] + wd2[(x, k, n)] + late_first[(x, k, n)] + Wp[(x, k, n)] for x in tasks for k in
            Vehicles for n in range(N)))

        # 求解模型并将输出重定向到文件
        with open('solver_log.txt', 'w') as f:
            with redirect_stdout(f):
                solution_status = mdl.solve(log_output=True)

                print("解的目标值:", mdl.objective_value)
                for x in tasks:
                    print(f"任务 {x} 的开始时间 Ts_x: {Ts_x[x].solution_value}")

        if solution_status:
            # 收集每辆车的时间点和相关事件
            objective_value = mdl.objective_value
            print(f"最小化的目标函数值: {objective_value}")
            vehicle_events = {k: [] for k in Vehicles}

            if solution_status:
                print("Solution found:")
                for x in tasks:
                    for k in Vehicles:
                        for n in range(N):
                            if mdl.solution.get_value(delivery_by_vehicle[(x, k, n)]) == 1:
                                vehicle_events[k].append({
                                    '时间': mdl.solution.get_value(Tk0[(x, k, n)]),
                                    '事件': f"{n} 趟次空闲，任务为{x}"
                                })
                                vehicle_events[k].append({
                                    '时间': mdl.solution.get_value(Tk1[(x, k, n)]),
                                    '事件': f"{n} 趟次发车"
                                })
                                vehicle_events[k].append({
                                    '时间': mdl.solution.get_value(Tk2[(x, k, n)]),
                                    '事件': f"{n} 趟次到达"
                                })
                                vehicle_events[k].append({
                                    '时间': mdl.solution.get_value(Tk3[(x, k, n)]),
                                    '事件': f"{n} 趟次浇筑开始"
                                })
                                vehicle_events[k].append({
                                    '时间': mdl.solution.get_value(Tk4[(x, k, n)]),
                                    '事件': f"{n} 趟次浇筑结束"
                                })
                                vehicle_events[k].append({
                                    '时间': mdl.solution.get_value(Tk5[(x, k, n)]),
                                    '事件': f"{n} 趟次返厂"
                                })

                # 按照时间排序并输出每辆车的事件
                for k in Vehicles:
                    print(f"车辆 {k} 的时间轴：")
                    vehicle_events[k].sort(key=lambda e: e['时间'])
                    for event in vehicle_events[k]:
                        print(f"时间: {event['时间']:.2f} - {event['事件']}")
                    print("-" * 40)  # 分隔线

                # 收集每个产线的生产批次时间信息
                line_events = {l: [] for l in lines}

                for x in tasks:
                    for l in lines:
                        for h in range(H):
                            if mdl.solution.get_value(production_by_line[(x, l, h)]) == 1:
                                line_events[l].append({
                                    '时间': mdl.solution.get_value(Ts_x[x]),
                                    '事件': f"生产批次 {h} 开始,任务为{x}"
                                })
                                line_events[l].append({
                                    '时间': mdl.solution.get_value(Tc_x[x, l, h]),
                                    '事件': f"生产批次 {h} 结束"
                                })

                # 按照时间排序并输出每个产线的生产批次时间信息
                for l in lines:
                    print(f"产线 {l} 的时间轴：")
                    line_events[l].sort(key=lambda e: e['时间'])
                    for event in line_events[l]:
                        print(f"时间: {event['时间']:.2f} - {event['事件']}")
                    print("-" * 40)  # 分隔线

                # 收集每个任务的详细信息
                task_details = {}

                for x in tasks:
                    task_details[x] = {
                        '浇筑开始时间': mdl.solution.get_value(Ts_g[x]),
                        '浇筑结束时间': mdl.solution.get_value(Te_g[x]),
                        '运输信息': [],
                        '生产信息': []
                    }

                    # 收集运输信息
                    for k in Vehicles:
                        for n in range(N):
                            if mdl.solution.get_value(delivery_by_vehicle[(x, k, n)]) == 1:
                                task_details[x]['运输信息'].append({
                                    '车辆': k,
                                    '趟次': n
                                })

                    # 收集生产信息
                    for l in lines:
                        for h in range(H):
                            if mdl.solution.get_value(production_by_line[(x, l, h)]) == 1:
                                task_details[x]['生产信息'].append({
                                    '产线': l,
                                    '批次': h
                                })

                # 输出每个任务的详细信息
                for x in tasks:
                    print(f"任务 {x} 的详细信息：")
                    print("运输信息:")
                    for transport in task_details[x]['运输信息']:
                        print(f"  - 车辆 {transport['车辆']} 的 {transport['趟次']} 趟次")
                    print("生产信息:")
                    for production in task_details[x]['生产信息']:
                        print(f"  - 产线 {production['产线']} 的 {production['批次']} 批次")

                    print("-" * 40)  # 分隔线
        else:
            print("No solution found.")

    except Exception as e:
        print(f"读取 Excel 文件时发生错误: {e}")