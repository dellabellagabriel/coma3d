% This script takes all preprocessed files ROI* (copied from the conn
% folder) and saves them into a time x roi matrix for each subject

clc,clear

n_subjects = 47;
n_rois = 34;
n_windows = 23;

path_to_project = '/home/usuario/disco1/proyectos/2023-Coma3D';
path_to_timeseries = [path_to_project, '/conn_project_coma3D_power300/results/firstlevel/DMN'];
path_to_output = [path_to_project, '/matrices_dmn'];

list_of_conditions = dir([path_to_timeseries, '/resultsROI_Subject*']);

counter = 0;
iSub = 1;
connectivity_list = zeros(n_subjects, n_windows, n_rois, n_rois);
for iCond=1:length(list_of_conditions)
    name = list_of_conditions(iCond).name;
    disp(name)
    load([path_to_timeseries, '/', name], 'Z')

    %We ignore the first file and the last two (because of the setup > conditions)
    if counter > 0 && counter <= n_windows
        connectivity_list(iSub, counter, :, :) = Z;
    end
    counter = counter + 1;

    %Every windows+3 files we begin with a new subject
    if counter == n_windows+3
        counter = 0;
        data = squeeze(connectivity_list(iSub, :, :, :));
        data(:, logical(eye(n_rois, n_rois))) = 1;
        data = single(data);
        save([path_to_output, '/', name(1:21), '.mat'], 'data')
        iSub = iSub + 1;
    end

end

disp('done')