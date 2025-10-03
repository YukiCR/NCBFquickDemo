classdef carModel < handle
    %% dynamics related
    properties
        % 小车的位置和方向
        R = 0.2; % 半径
        px = 0; % x 轴位置
        py = 0; % y 轴位置
        theta = 0; % 方向角
        target = [0,0,0]; % 目标点
        dt = 0.1; % 时间步长 (默认为 0.1)
        T = 1000;
        history = []; % 状态历史记录 (n x 3 矩阵)
        hisCnt = 1;
        isRecordHistory = true;% 是否记录历史状态
        lastInput = []; % 上一个输入， 2x1
    end

    methods
        % 构造函数初始化位置、角度、时间步长和记录开关
        function obj = carModel(px_init, py_init, theta_init, target_init, dt_init, isRecordHistory_init, T)
            if nargin > 0
                obj.px = px_init;
                obj.py = py_init;
                obj.theta = theta_init;
                obj.target = target_init;
            end
            if nargin > 4
                obj.dt = dt_init;
            end
            if nargin > 5
                obj.isRecordHistory = isRecordHistory_init;
            end
            if nargin > 6
                obj.T = T;
            end

            % 如果开启了历史记录功能，初始化时也记录状态
            if obj.isRecordHistory
                obj.history = nan(obj.T, 3);
                obj.recordState();
                obj.lastInput = [0;0];
            end
        end

        % 更新小车状态的方法
        function updateState(obj, in1, in2)
            if nargin == 3
                v = in1;
                omega = in2;
            end
            if nargin == 2 % vector form
                v = in1(1);
                omega = in1(2);
            end
            % v：速度，omega：角速度
            % 更新方向角
            obj.theta = obj.theta + omega * obj.dt;
            % 更新位置
            obj.px = obj.px + v * cos(obj.theta) * obj.dt;
            obj.py = obj.py + v * sin(obj.theta) * obj.dt;

            % 如果开启了历史记录功能，更新后记录状态
            if obj.isRecordHistory
                obj.recordState();
                obj.lastInput = [v; omega];
            end
        end

        % 获取当前状态的方法
        function [px, py, theta] = getState(obj)
            px = obj.px;
            py = obj.py;
            theta = obj.theta;
        end

        % 记录当前状态的方法
        function recordState(obj)
            % 将当前状态存入 history (一行数据：px, py, theta)
            state = [obj.px, obj.py, obj.theta];
            obj.history(obj.hisCnt, :) = state;
            obj.hisCnt = obj.hisCnt+1;
        end

        % 获取历史记录的方法
        function history = getHistory(obj)
            history = obj.history;
        end
    end

    %% nominal control related
    properties
        KP = [0.1, 1];
    end

    methods
        function [v,omega] = getPcontrol(obj)
            % see also
            % CONTROL OF UNICYCLE TYPE ROBOTS
            % Tracking, Path Following and Point Stabilization
            errVec = [obj.target(1) - obj.px; obj.target(2) - obj.py];
            % 计算位置误差
            distance_error = errVec' * [cos(obj.theta); sin(obj.theta)];
            % 计算角度误差
            deltaTheta = wrapToPi(atan2(errVec(2), errVec(1)) - obj.theta);
            angle_error = deltaTheta;

            % 根据比例控制来计算线速度和角速度
            v = obj.KP(1) * distance_error;  % 线速度与位置误差成正比
            omega = obj.KP(2) * angle_error; % 角速度与角度误差成正比
        end

        function [v,omega] = getPorpotionalcontrol(obj)
            D = 0.1;
            [pax, pay, patheta] = obj.getState();
            Pa = [pax + D*cos(patheta); pay + D*sin(patheta)];
            % matrix transforming the system to fully actuated system
            M = [cos(patheta), -D*sin(patheta);
                sin(patheta), D*cos(patheta)];
            
            goal = [obj.target(1); obj.target(2)];
            V = (goal - Pa) * 0.1;
            vs = M \ V;

            v = vs(1);
            omega = vs(2);
        end
    end

    %% R-CBF related
    properties
        alpha = 0.3;
    end

    methods
        function filteredInput = RCBF_Filter(obj, agentArr, nominalInput)
            D = 0.1;
            n = length(agentArr);

            % get states
            [pax, pay, patheta] = obj.getState();
            Pa = [pax; pay];
            % matrix transforming the system to fully actuated system
            M = [cos(patheta), -D*sin(patheta);
                sin(patheta), D*cos(patheta)];

            A = zeros(n, 2);
            B = zeros(n, 1);
            % iterate to get constraints
            for i = 1:n
                [pbx, pby, pbtheta] = agentArr{i}.getState();
                Pb = [pbx; pby];
                Pm = 0.5 * (Pa + Pb); % middle point
                na_hat = (Pa - Pm)/norm(Pa - Pm);
                b = 0.5*obj.alpha*(norm(Pa - Pb) - 2*obj.R);
                A(i, :) = -na_hat' * M;
                B(i) = b;
            end

            opt = optimoptions("quadprog","Display","off");

            % NOTE: THE BIG M REALLLLY MATTERS HERE
            % H = 2 * eye(2);
            % f = -2 * M * nominalInput;
            % [filteredInput, ~, flag, ~] = quadprog(H,f,-na_hat',B,[],[],[],[],[],opt);
            % filteredInput = M \ filteredInput;

            H = 2 * M' * eye(2) * M;
            f = -2 * (M' * M) * nominalInput;
            [filteredInput, ~, flag, ~] = quadprog(H, f, A, B,[],[],[],[],[],opt);

            if flag < 0
                warning("QP error, using minimum invasive result")
                filteredInput = mimaxlinear(A, B);
            end
        end

        function filteredInput = RCBF_Filter_circulation(obj, agentArr, nominalInput)
            D = 0.1;
            Omega = [0 -1;
                1 0];
            circ_a = 3;
            circ_b = 0.1;
            n = length(agentArr);

            % get states
            [pax, pay, patheta] = obj.getState();
            Pa = [pax; pay];
            % matrix transforming the system to fully actuated system
            M = [cos(patheta), -D*sin(patheta);
                sin(patheta), D*cos(patheta)];

            % cbf constraints
            A_cbf = zeros(n, 2);
            B_cbf = zeros(n, 1);
            % circulation constraints
            A_circ = zeros(n, 2);
            B_circ = zeros(n, 1);
            % iterate to get constraints
            for i = 1:n
                [pbx, pby, pbtheta] = agentArr{i}.getState();
                Pb = [pbx; pby];
                Pm = 0.5 * (Pa + Pb); % middle point
                na_hat = (Pa - Pm)/norm(Pa - Pm);
                d = norm(Pa - Pb) - 2*obj.R;
                b = 0.5*obj.alpha*d;
                A_cbf(i, :) = -na_hat' * M;
                B_cbf(i) = b;

                T = Omega * na_hat;
                A_circ(i, :) = -T' * M;
                B_circ(i, :) = -circ_a + circ_b*d;
            end

            A = [A_cbf; A_circ];
            B = [B_cbf; B_circ];

            opt = optimoptions("quadprog","Display","off");

            H = 2 * M' * eye(2) * M;
            f = -2 * (M' * M) * nominalInput;
            [filteredInput, ~, flag, ~] = quadprog(H, f, A, B,[],[],[],[],[],opt);

            if flag < 0 % if fall, use safety constraints only
                % warning("QP error, using safety constraints only")
                [filteredInput, ~, flag, ~] = quadprog(H, f, A_cbf, B_cbf,[],[],[],[],[],opt);
                % if fall as well, use minmaxd instead
                if flag < 0
                    % warning("QP error, using minimum invasive result")
                    filteredInput = minmaxlinear(A_cbf, B_cbf);
                end
            end
        end

        function filteredInput = RCBF_Filter_nearest_circulation(obj, agentArr, idxNow, nominalInput)
            D = 0.1;
            Omega = [0 -1;
                1 0];
            circ_a = 0.5;
            circ_b = 0.8;
            n = length(agentArr)-1;

            % get states
            [pax, pay, patheta] = obj.getState();
            Pa = [pax; pay];
            % matrix transforming the system to fully actuated system
            M = [cos(patheta), -D*sin(patheta);
                sin(patheta), D*cos(patheta)];

            % cbf constraints
            A_cbf = zeros(n, 2);
            B_cbf = zeros(n, 1);
            % iterate to get constraints
            dnearest = +inf;
            na_hat_nearest = [0;0];
            cnt = 1;
            for i = 1:n+1
                if i == idxNow
                    continue
                end
                [pbx, pby, pbtheta] = agentArr{i}.getState();
                Pb = [pbx; pby];
                Pm = 0.5 * (Pa + Pb); % middle point
                na_hat = (Pa - Pm)/norm(Pa - Pm);
                d = norm(Pa - Pb) - 2*obj.R;
                b = 0.5*obj.alpha*d;
                A_cbf(cnt, :) = -na_hat' * M;
                B_cbf(cnt) = b;
                cnt = cnt + 1;
                
                if d < dnearest
                    dnearest = d;
                    na_hat_nearest = na_hat;
                end
            end

            A_circ = -(Omega * na_hat_nearest)' * M;
            % B_circ = -circ_a + circ_b*dnearest;
            % p_i = [-4.00000000000000	6.00000000000000	-3.00000000000000	0.500000000000000]';
            p_i = [-0.585937500000000	1.40625000000000	-1.12500000000000	0.300000000000000]';
            B_circ = - [dnearest^3, dnearest^2, dnearest, 1] * p_i;

            A = [A_cbf; A_circ];
            B = [B_cbf; B_circ];

            opt = optimoptions("quadprog","Display","off");

            H = 2 * M' * eye(2) * M;
            f = -2 * (M' * M) * nominalInput;
            [filteredInput, ~, flag, ~] = quadprog(H, f, A, B,[],[],[],[],[],opt);

            if flag < 0 % if fall, use safety constraints only
                % warning("QP error, using safety constraints only")
                [filteredInput, ~, flag, ~] = quadprog(H, f, A_cbf, B_cbf,[],[],[],[],[],opt);
                % if fall as well, use minmaxd instead
                if flag < 0
                    % warning("QP error, using minimum invasive result")
                    filteredInput = minmaxlinear(A_cbf, B_cbf);
                end
            end
        end


        function filteredInput = RCBF_Filter_nearest_circulation_soft(obj, agentArr, idxNow, nominalInput)
            D = 0.1;
            Omega = [0 -1;
                1 0];
            circ_a = 0.5;
            circ_b = 0.8;
            n = length(agentArr)-1;

            % get states
            [pax, pay, patheta] = obj.getState();
            Pa = [pax; pay];
            % matrix transforming the system to fully actuated system
            M = [cos(patheta), -D*sin(patheta);
                sin(patheta), D*cos(patheta)];

            % cbf constraints
            A_cbf = zeros(n, 2+1);
            B_cbf = zeros(n, 1);
            % iterate to get constraints
            dnearest = +inf;
            na_hat_nearest = [0;0];
            cnt = 1;
            for i = 1:n+1
                if i == idxNow
                    continue
                end
                [pbx, pby, pbtheta] = agentArr{i}.getState();
                Pb = [pbx; pby];
                Pm = 0.5 * (Pa + Pb); % middle point
                na_hat = (Pa - Pm)/norm(Pa - Pm);
                d = norm(Pa - Pb) - 2*obj.R;
                b = 0.5*obj.alpha*d;
                A_cbf(cnt, :) = [-na_hat' * M, 0];
                B_cbf(cnt) = b;
                cnt = cnt + 1;
                
                if d < dnearest
                    dnearest = d;
                    na_hat_nearest = na_hat;
                end
            end

            A_circ = [-(Omega * na_hat_nearest)' * M, -1];
            % B_circ = -circ_a + circ_b*dnearest;
            % p_i = [-4.00000000000000	6.00000000000000	-3.00000000000000	0.500000000000000]';
            p_i = [-0.585937500000000	1.40625000000000	-1.12500000000000	0.300000000000000]';
            B_circ = - [dnearest^3, dnearest^2, dnearest, 1] * p_i;

            A = [A_cbf; A_circ];
            B = [B_cbf; B_circ];

            opt = optimoptions("quadprog","Display","off");
            
            M_extend = [M, zeros(2, 1);
                        zeros(1, 2), 1];
            nominalInput_extend = [nominalInput; 0];
            Q = [eye(2), zeros(2, 1); zeros(1, 2), 5];
            H = 2 * M_extend' * Q * M_extend;
            f = -2 * (M_extend' * M_extend) * nominalInput_extend;
            [filteredInput, ~, flag, ~] = quadprog(H, f, A, B,[],[],[],[],[],opt);

            if flag < 0 % if fall, use safety constraints only
                warning("QP error, using safety constraints only")
                [filteredInput, ~, flag, ~] = quadprog(H, f, A_cbf, B_cbf,[],[],[],[],[],opt);
                % if fall as well, use minmaxd instead
                if flag < 0
                    warning("QP error, using minimum invasive result")
                    filteredInput = minmaxlinear(A_cbf, B_cbf);
                end
            end
        end

        


    end

    %% visualization related
    methods
        function plotTraj(obj)
            plot(obj.history(:,1), obj.history(:,2));
        end

        function plotHistory(obj)
            stepInterval = 30;  % 默认每stepInterval步绘制一次

            % 绘制轨迹
            p = plot(obj.history(1:stepInterval:end,1), obj.history(1:stepInterval:end,2), "LineWidth", 1, "LineStyle","--","Marker","o", "MarkerSize",8);

            xs = nan(length(1:stepInterval:length(obj.history)-1),1);
            ys = nan(length(1:stepInterval:length(obj.history)-1),1);
            vxs = nan(length(1:stepInterval:length(obj.history)-1),1);
            vys = nan(length(1:stepInterval:length(obj.history)-1),1);
            cnt = 1;

            for i = 1:stepInterval:length(obj.history)-1
                % 获取当前点的坐标和方向
                x = obj.history(i, 1);
                y = obj.history(i, 2);
                angle = obj.history(i, 3);

                % 计算速度矢量（假设速度为常数v）
                v = 1.0;  % 假设恒定速度为 1 m/s
                vx = v * cos(angle);  % x 方向速度
                vy = v * sin(angle);  % y 方向速度

                xs(cnt) = x;
                ys(cnt) = y;
                vxs(cnt) = vx;
                vys(cnt) = vy;

                cnt = cnt + 1;
            end

            % 使用quiver绘制速度箭头
            quiver(xs, ys, vxs, vys, 0.5, 'LineWidth', 1.0, 'MaxHeadSize', 1.0, "Color", p.Color);
        end

    end
end
