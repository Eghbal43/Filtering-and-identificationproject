% Extended Kalman Filter Simulation for an Autonomous Robot

% Time settings
dt = 0.1; % time step
T = 60; % total time
time_vector = 0:dt:T-dt;

% Initial state [x; y; orientation (phi); velocity (v)]
x_est = [0; 0; 0; 1]; % example initial state
P = eye(4); % initial state covariance matrix
Q = diag([0.01, 0.01, 0.01, 0.01]); % process noise covariance matrix
R = diag([0.1, 0.1]); % observation noise covariance matrix for speed and gyro
K_store = cell(1, length(time_vector)); % Initialize a cell array to store Kalman gain matrices
% Initialize an array to store innovation terms
innovation_store = zeros(length(time_vector), 2); % Assuming 2-dimensional measurements

% Storage for estimated states
est_states = zeros(4, length(time_vector));

% Simulate true movement and noisy measurements
true_states = zeros(4, length(time_vector));
measurements = zeros(2, length(time_vector));

% Assuming the robot moves in a circle with a radius of 20 meters
radius = 20;
omega = 1; % constant angular velocity in rad/s
for k = 2:length(time_vector)
    % Simulate the true state
    true_states(1,k) = radius * cos(omega * time_vector(k));
    true_states(2,k) = radius * sin(omega * time_vector(k));
    true_states(3,k) = omega * time_vector(k);
    true_states(4,k) = omega * radius; % constant velocity
    
    % Simulate the noisy measurements (speed and gyro)
    speed_noise = sqrt(R(1,1)) * randn;
    gyro_noise = sqrt(R(2,2)) * randn;
    measurements(:,k) = [true_states(4,k) + speed_noise; true_states(3,k) + gyro_noise];
end

% EKF
for k = 1:length(time_vector)
    % Prediction step
    F = [1, 0, -x_est(4)*sin(x_est(3))*dt, cos(x_est(3))*dt;
         0, 1, x_est(4)*cos(x_est(3))*dt, sin(x_est(3))*dt;
         0, 0, 1, 0;
         0, 0, 0, 1]; % Jacobian of the motion model
    x_pred = x_est + [x_est(4)*cos(x_est(3))*dt; 
                      x_est(4)*sin(x_est(3))*dt; 
                      omega*dt; % Orientation changes with constant omega
                      0]; % assuming constant velocity
    P = F*P*F' + Q;

    % Update step
    % Measurement matrix
    H = [0, 0, 0, 1; % speed measurement affects only the velocity
         0, 0, 1, 0]; % gyro measurement affects only the orientation
    
    % Kalman gain
    K = P*H'/(H*P*H' + R);
    K_store{k} = K;
    
    % Measurement update
    z = measurements(:,k); % Your actual measurement
    y = z - H*x_pred; % Innovation term
    x_est = x_pred + K*y;
    P = (eye(4) - K*H)*P;

    % Store the innovation term
    innovation_store(k, :) = y;
    
    % Save the state estimates
    est_states(:,k) = x_est;
end

% ... (previous code)

% Plotting results
figure;

% Plot for X position
subplot(4,1,1);
plot(time_vector, true_states(1,:), 'g', time_vector, est_states(1,:), 'b');
title('X Position');
xlabel('Time (s)');
ylabel('X (m)');
legend('True', 'Estimated');

% Plot for Y position
subplot(4,1,2);
plot(time_vector, true_states(2,:), 'g', time_vector, est_states(2,:), 'b');
title('Y Position');
xlabel('Time (s)');
ylabel('Y (m)');
legend('True', 'Estimated');

% Plot for Orientation (phi)
subplot(4,1,3);
plot(time_vector, true_states(3,:), 'g', time_vector, est_states(3,:), 'b', time_vector, measurements(2,:), 'r:');
title('Orientation (phi)');
xlabel('Time (s)');
ylabel('Phi (rad)');
legend('True', 'Estimated', 'Measured');

% Plot for Velocity (v)
subplot(4,1,4);
plot(time_vector, true_states(4,:), 'g', time_vector, est_states(4,:), 'b', time_vector, measurements(1,:), 'r:');
title('Velocity (v)');
xlabel('Time (s)');
ylabel('Velocity (m/s)');
legend('True', 'Estimated', 'Measured');

% Adjust the figure properties for better visualization
set(gcf, 'Position', [100, 100, 720, 800]); % Resize the figure window
sgtitle('Extended Kalman Filter Results'); % Add a title to the figure

% Plotting Kalman Gain values
num_elements = numel(K_store{1});
figure;
for i = 1:num_elements
    subplot(num_elements, 1, i);
    plot(cellfun(@(x) x(i), K_store));
    title(['Kalman Gain Element ', num2str(i)]);
    xlabel('Time Step');
    ylabel('Gain Value');
end
% Assuming you have a cell array 'K_store' with Kalman gain matrices

% Increase figure size
figure;
set(gcf, 'Position', [100, 100, 800, 600]); % Adjust [left, bottom, width, height] as needed

% Use a tiled layout for better control (MATLAB R2019b and later)
t = tiledlayout(4, 2); % Adjust number of rows and columns as needed
title(t, 'Kalman Gain Elements Over Time');

% Loop through each element of the Kalman gain
for i = 1:8
    nexttile; % Moves to the next tile in the layout
    plot(cellfun(@(x) x(i), K_store));
    xlabel('Time Step');
    ylabel(['K(', num2str(i), ')']);
end

% Adjust padding between plots
t.Padding = 'compact';
t.TileSpacing = 'compact';

% Plotting Innovation terms
figure;
subplot(2,1,1);
plot(time_vector, innovation_store(:,1));
title('Innovation Term for First Measurement');
xlabel('Time (s)');
ylabel('Innovation');

subplot(2,1,2);
plot(time_vector, innovation_store(:,2));
title('Innovation Term for Second Measurement');
xlabel('Time (s)');
ylabel('Innovation');


