import wave_1d_fd_tf.run_timing

t=wave_1d_fd_tf.run_timing.run_timing_num_steps()
t.to_csv('times_num_steps.csv')

t=wave_1d_fd_tf.run_timing.run_timing_model_size()
t.to_csv('times_model_size.csv')

#t=wave_1d_fd_tf.run_timing.run_timing_model_size(num_steps=2000, model_sizes=[2000])
#t.to_csv('times_2000_2000.csv')
