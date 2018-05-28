rsync -avhp jhansen@erehwon.cim.mcgill.ca:///localdata/jhansen/trajectories/examples/all_results_model_vqvae_pcnn_model_rollouts_50_length_5_prior_goal.pkl .

echo 'finished erehwon 50/5'
rsync -avhp jhansen@roke.cim.mcgill.ca:///localdata/jhansen/trajectories/examples/all_results_model_vqvae_pcnn_model_rollouts_50_length_10_prior_goal.pkl .
echo 'finished roke 50/10'

rsync -avhp jhansen@numenor.cim.mcgill.ca:///localdata/jhansen/trajectories/examples/all_results_model_vqvae_pcnn_model_rollouts_100_length_5_prior_goal.pkl .
echo 'finished numenor 100/5'

rsync -avhp jhansen@raza.cim.mcgill.ca:///localdata/jhansen/trajectories/examples/all_results_model_vqvae_pcnn_model_rollouts_100_length_10_prior_goal.pkl .
echo 'finished raza 100/10'
