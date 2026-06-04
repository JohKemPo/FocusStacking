#include "stack/laplacian_pyr.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace fs_core
{

    LaplacianPyramidStacker::LaplacianPyramidStacker(int depth, int kernel_size)
        : depth_(depth), kernel_size_(kernel_size), is_first_image_(true) {}

    void LaplacianPyramidStacker::generatePyramid(const ImageType &img, std::vector<cv::UMat> &lap_pyr, cv::UMat &base)
    {
        cv::UMat curr_img;
        img.convertTo(curr_img, CV_32F);

        for (int i = 0; i < depth_ - 1; ++i)
        {
            cv::UMat down, up, lap;
            cv::pyrDown(curr_img, down);
            cv::pyrUp(down, up, curr_img.size());
            cv::subtract(curr_img, up, lap);

            lap_pyr.push_back(std::move(lap));
            up.release();

            curr_img = std::move(down);
        }
        lap_pyr.push_back(curr_img);
        base = curr_img;
    }

    cv::UMat LaplacianPyramidStacker::calculateDeviation(const cv::UMat &src) const
    {
        cv::UMat gray_src;
        if (src.channels() > 1)
        {
            cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray_src = src;
        }

        cv::UMat src_sq, mean, mean_sq, sq_mean, variance;

        cv::multiply(gray_src, gray_src, src_sq);
        cv::boxFilter(gray_src, mean, CV_32F, cv::Size(kernel_size_, kernel_size_), cv::Point(-1, -1), true, cv::BORDER_REFLECT101);
        cv::boxFilter(src_sq, sq_mean, CV_32F, cv::Size(kernel_size_, kernel_size_), cv::Point(-1, -1), true, cv::BORDER_REFLECT101);

        src_sq.release();

        cv::multiply(mean, mean, mean_sq);
        cv::subtract(sq_mean, mean_sq, variance);

        mean.release();
        mean_sq.release();
        sq_mean.release();

        cv::UMat stddev;
        cv::max(variance, 0.0, variance);
        cv::sqrt(variance, stddev);
        return stddev;
    }

    cv::UMat LaplacianPyramidStacker::calculateRegionEnergy(const cv::UMat &src) const
    {
        cv::UMat gray_src;
        if (src.channels() > 1)
        {
            cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray_src = src;
        }

        cv::UMat src_sq, energy;
        cv::multiply(gray_src, gray_src, src_sq);

        gray_src.release();

        cv::Mat kernel = (cv::Mat_<float>(5, 5) << 0.0025, 0.0125, 0.0200, 0.0125, 0.0025,
                          0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
                          0.0200, 0.1000, 0.1600, 0.1000, 0.0200,
                          0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
                          0.0025, 0.0125, 0.0200, 0.0125, 0.0025);

        cv::filter2D(src_sq, energy, CV_32F, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);
        src_sq.release();
        return energy;
    }

    void LaplacianPyramidStacker::addImage(const ImageType &img)
    {
        std::vector<cv::UMat> current_lap_pyr;
        cv::UMat current_base;

        generatePyramid(img, current_lap_pyr, current_base);

        if (is_first_image_)
        {
            fused_lap_pyr_ = std::move(current_lap_pyr);
            fused_base_ = std::move(current_base);

            max_deviation_map_ = calculateDeviation(fused_lap_pyr_.back());
            for (int l = 0; l < depth_ - 1; ++l)
            {
                max_region_energy_.push_back(calculateRegionEnergy(fused_lap_pyr_[l]));
            }
            is_first_image_ = false;
            return;
        }

        cv::UMat curr_dev = calculateDeviation(current_lap_pyr.back());
        cv::UMat dev_mask;
        cv::compare(curr_dev, max_deviation_map_, dev_mask, cv::CMP_GT);

        current_lap_pyr.back().copyTo(fused_lap_pyr_.back(), dev_mask);
        curr_dev.copyTo(max_deviation_map_, dev_mask);

        curr_dev.release();
        dev_mask.release();

        for (int l = 0; l < depth_ - 1; ++l)
        {
            cv::UMat curr_re = calculateRegionEnergy(current_lap_pyr[l]);
            cv::UMat re_mask;

            cv::compare(curr_re, max_region_energy_[l], re_mask, cv::CMP_GT);

            current_lap_pyr[l].copyTo(fused_lap_pyr_[l], re_mask);
            curr_re.copyTo(max_region_energy_[l], re_mask);

            curr_re.release();
            re_mask.release();
        }

        for (auto &level : current_lap_pyr)
            level.release();
        current_base.release();
    }

    ImageType LaplacianPyramidStacker::finalize()
    {
        if (fused_lap_pyr_.empty())
        {
            throw std::runtime_error("Nenhuma imagem adicionada ao stacker.");
        }

        cv::UMat reconstructed = fused_lap_pyr_.back();
        for (int i = depth_ - 2; i >= 0; --i)
        {
            cv::UMat upsampled;
            cv::pyrUp(reconstructed, upsampled, fused_lap_pyr_[i].size());
            cv::add(upsampled, fused_lap_pyr_[i], reconstructed, cv::noArray(), CV_32F);
            upsampled.release();
        }

        ImageType final_result;
        reconstructed.convertTo(final_result, CV_8UC3);
        return final_result;
    }

} 