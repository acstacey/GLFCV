/**
 * This code has been adapted from:
 * https://github.com/atilimcetin/guided-filter
 * Copyright (c) 2014 Atilim Cetin
 * to use the OpenCV CUDA API.
 *
 * It implements the guided filter by Kaiming He (http://kaiminghe.com/eccv10/)
 *
 */

/**
 * LF-TADCG-CUDA-DISP - Code for estimating the disparity map of a light field
 *
 * Copyright (C) 2017 Adam Stacey
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "guidedfilter.h"

static cv::cuda::GpuMat convertTo(const cv::cuda::GpuMat &mat, int depth) {
  if (mat.depth() == depth)
    return mat;

  cv::cuda::GpuMat result;
  mat.convertTo(result, depth);
  return result;
}

class GuidedFilterImpl {
 public:
  virtual ~GuidedFilterImpl() {}

  cv::cuda::GpuMat filter(const cv::cuda::GpuMat &p, int depth, cv::cuda::Stream &stream);

 protected:
  int Idepth;

 private:
  virtual cv::cuda::GpuMat filterSingleChannel(const cv::cuda::GpuMat &p,
                                               cv::cuda::Stream &stream = cv::cuda::Stream::Null()) const = 0;
};

class GuidedFilterMono : public GuidedFilterImpl {
 public:
  GuidedFilterMono(const cv::cuda::GpuMat &I, int r, double eps);

 private:
  virtual cv::cuda::GpuMat filterSingleChannel(const cv::cuda::GpuMat &p,
                                               cv::cuda::Stream &stream = cv::cuda::Stream::Null()) const;

 private:
  int r;
  double eps;
  cv::Ptr<cv::cuda::Filter> box_filter;
  cv::cuda::GpuMat I, mean_I, var_I;
};

class GuidedFilterColor : public GuidedFilterImpl {
 public:
  GuidedFilterColor(const cv::cuda::GpuMat &I, int r, double eps);

 private:
  virtual cv::cuda::GpuMat filterSingleChannel(const cv::cuda::GpuMat &p,
                                               cv::cuda::Stream &stream = cv::cuda::Stream::Null()) const;

 private:
  std::vector<cv::cuda::GpuMat> Ichannels;
  int r;
  double eps;
  cv::Ptr<cv::cuda::Filter> box_filter;
  cv::cuda::GpuMat mean_I_r, mean_I_g, mean_I_b;
  cv::cuda::GpuMat invrr, invrg, invrb, invgg, invgb, invbb;
};

cv::cuda::GpuMat GuidedFilterImpl::filter(const cv::cuda::GpuMat &p,
                                          int depth, cv::cuda::Stream &stream = cv::cuda::Stream::Null()) {
  cv::cuda::GpuMat p2 = convertTo(p, Idepth);

  cv::cuda::GpuMat result;
  if (p.channels() == 1) {
    result = filterSingleChannel(p2);
  } else {
    std::vector<cv::cuda::GpuMat> pc;
    cv::cuda::split(p2, pc);

    for (std::size_t i = 0; i < pc.size(); ++i)
      pc[i] = filterSingleChannel(pc[i]);

    cv::cuda::merge(pc, result);
  }

  return convertTo(result, depth == -1 ? p.depth() : depth);
}

GuidedFilterMono::GuidedFilterMono(const cv::cuda::GpuMat &origI,
                                   int r, double eps) : r(r), eps(eps) {

  if (origI.depth() == CV_32F || origI.depth() == CV_64F)
    I = origI.clone();
  else
    I = convertTo(origI, CV_32F);

  Idepth = I.depth();

  box_filter = cv::cuda::createBoxFilter(I.type(), I.type(), cv::Size(r, r));

  box_filter->apply(I, mean_I);
  cv::cuda::GpuMat mean_II;
  cv::cuda::multiply(I, I, mean_II);
  box_filter->apply(mean_II, mean_II);
  cv::cuda::multiply(mean_I, mean_I, var_I);
  cv::cuda::subtract(mean_II, var_I, var_I);
}

cv::cuda::GpuMat GuidedFilterMono::filterSingleChannel(const cv::cuda::GpuMat &p, cv::cuda::Stream &stream) const {
  cv::cuda::GpuMat mean_p, mean_Ip, cov_Ip;
  box_filter->apply(p, mean_p, stream);
  cv::cuda::multiply(I, p, mean_Ip, 1, -1, stream);
  box_filter->apply(mean_Ip, mean_Ip, stream);
  cv::cuda::multiply(mean_I, mean_p, cov_Ip, 1, -1, stream);
  cv::cuda::subtract(mean_Ip,
                     cov_Ip,
                     cov_Ip,
                     cv::noArray(),
                     -1,
                     stream); // this is the covariance of (I, p) in each local patch.

  cv::cuda::GpuMat a, b;
  cv::cuda::add(var_I, cv::Scalar(eps), a, cv::noArray(), -1, stream);
  cv::cuda::divide(cov_Ip, a, a, 1, -1, stream); // Eqn. (5) in the paper;

  cv::cuda::multiply(a, mean_I, b, 1, -1, stream);
  cv::cuda::subtract(mean_p, b, b, cv::noArray(), -1, stream); // Eqn. (6) in the paper;

  box_filter->apply(a, a, stream);
  box_filter->apply(b, b, stream);

  cv::cuda::multiply(a, I, a, 1, -1, stream);
  cv::cuda::add(a, b, a, cv::noArray(), -1, stream);

  return a;
}

GuidedFilterColor::GuidedFilterColor(const cv::cuda::GpuMat &origI,
                                     int r,
                                     double eps) : r(r), eps(eps) {
  cv::cuda::GpuMat I;
  if (origI.depth() == CV_32F || origI.depth() == CV_64F)
    I = origI.clone();
  else
    I = convertTo(origI, CV_32F);

  Idepth = I.depth();
  box_filter = cv::cuda::createBoxFilter(CV_32FC1, CV_32FC1, cv::Size(r, r));

  cv::cuda::split(I, Ichannels);

  box_filter->apply(Ichannels[0], mean_I_r);
  box_filter->apply(Ichannels[1], mean_I_g);
  box_filter->apply(Ichannels[2], mean_I_b);

  // variance of I in each local patch: the matrix Sigma in Eqn (14).
  // Note the variance in each local patch is a 3x3 symmetric matrix:
  //           rr, rg, rb
  //   Sigma = rg, gg, gb
  //           rb, gb, bb
  cv::cuda::GpuMat var_I_rr, var_I_rg, var_I_rb, var_I_gg, var_I_gb, var_I_bb, working;
  cv::cuda::multiply(Ichannels[0], Ichannels[0], working);
  box_filter->apply(working, var_I_rr);
  cv::cuda::multiply(mean_I_r, mean_I_r, working);
  cv::cuda::subtract(var_I_rr, working, var_I_rr);
  cv::cuda::add(var_I_rr, cv::Scalar(eps), var_I_rr);

  cv::cuda::multiply(Ichannels[0], Ichannels[1], working);
  box_filter->apply(working, var_I_rg);
  cv::cuda::multiply(mean_I_r, mean_I_g, working);
  cv::cuda::subtract(var_I_rg, working, var_I_rg);

  cv::cuda::multiply(Ichannels[0], Ichannels[2], working);
  box_filter->apply(working, var_I_rb);
  cv::cuda::multiply(mean_I_r, mean_I_b, working);
  cv::cuda::subtract(var_I_rb, working, var_I_rb);

  cv::cuda::multiply(Ichannels[1], Ichannels[1], working);
  box_filter->apply(working, var_I_gg);
  cv::cuda::multiply(mean_I_g, mean_I_g, working);
  cv::cuda::subtract(var_I_gg, working, var_I_gg);
  cv::cuda::add(var_I_gg, cv::Scalar(eps), var_I_gg);

  cv::cuda::multiply(Ichannels[1], Ichannels[2], working);
  box_filter->apply(working, var_I_gb);
  cv::cuda::multiply(mean_I_g, mean_I_b, working);
  cv::cuda::subtract(var_I_gb, working, var_I_gb);

  cv::cuda::multiply(Ichannels[2], Ichannels[2], working);
  box_filter->apply(working, var_I_bb);
  cv::cuda::multiply(mean_I_b, mean_I_b, working);
  cv::cuda::subtract(var_I_bb, working, var_I_bb);
  cv::cuda::add(var_I_bb, cv::Scalar(eps), var_I_bb);


  // Inverse of Sigma + eps * I
  cv::cuda::multiply(var_I_gg, var_I_bb, invrr);
  cv::cuda::multiply(var_I_gb, var_I_gb, working);
  cv::cuda::subtract(invrr, working, invrr);

  cv::cuda::multiply(var_I_gb, var_I_rb, invrg);
  cv::cuda::multiply(var_I_rg, var_I_bb, working);
  cv::cuda::subtract(invrg, working, invrg);

  cv::cuda::multiply(var_I_rg, var_I_gb, invrb);
  cv::cuda::multiply(var_I_gg, var_I_rb, working);
  cv::cuda::subtract(invrb, working, invrb);

  cv::cuda::multiply(var_I_rr, var_I_bb, invgg);
  cv::cuda::multiply(var_I_rb, var_I_rb, working);
  cv::cuda::subtract(invgg, working, invgg);

  cv::cuda::multiply(var_I_rb, var_I_rg, invgb);
  cv::cuda::multiply(var_I_rr, var_I_gb, working);
  cv::cuda::subtract(invgb, working, invgb);

  cv::cuda::multiply(var_I_rr, var_I_gg, invbb);
  cv::cuda::multiply(var_I_rg, var_I_rg, working);
  cv::cuda::subtract(invbb, working, invbb);

  cv::cuda::GpuMat covDet;
  cv::cuda::multiply(invrr, var_I_rr, var_I_rr);
  cv::cuda::multiply(invrg, var_I_rg, var_I_rg);
  cv::cuda::multiply(invrb, var_I_rb, var_I_rb);
  cv::cuda::add(var_I_rr, var_I_rg, covDet);
  cv::cuda::add(covDet, var_I_rb, covDet);

  cv::cuda::divide(invrr, covDet, invrr);
  cv::cuda::divide(invrg, covDet, invrg);
  cv::cuda::divide(invrb, covDet, invrb);
  cv::cuda::divide(invgg, covDet, invgg);
  cv::cuda::divide(invgb, covDet, invgb);
  cv::cuda::divide(invbb, covDet, invbb);
}

cv::cuda::GpuMat GuidedFilterColor::filterSingleChannel(const cv::cuda::GpuMat &p, cv::cuda::Stream &stream) const {
  cv::cuda::GpuMat mean_p, mean_Ip_r, mean_Ip_g, mean_Ip_b;
  cv::cuda::GpuMat cov_Ip_r, cov_Ip_g, cov_Ip_b, a_r, a_g, a_b, b;

  box_filter->apply(p, mean_p, stream);

  cv::cuda::multiply(Ichannels[0], p, mean_Ip_r, 1, -1, stream);
  cv::cuda::multiply(Ichannels[1], p, mean_Ip_g, 1, -1, stream);
  cv::cuda::multiply(Ichannels[2], p, mean_Ip_b, 1, -1, stream);
  box_filter->apply(mean_Ip_r, mean_Ip_r, stream);
  box_filter->apply(mean_Ip_g, mean_Ip_g, stream);
  box_filter->apply(mean_Ip_b, mean_Ip_b, stream);

  // covariance of (I, p) in each local patch.
  cv::cuda::multiply(mean_I_r, mean_p, cov_Ip_r, 1, -1, stream);
  cv::cuda::subtract(mean_Ip_r, cov_Ip_r, cov_Ip_r, cv::noArray(), -1, stream);
  cv::cuda::multiply(mean_I_g, mean_p, cov_Ip_g, 1, -1, stream);
  cv::cuda::subtract(mean_Ip_g, cov_Ip_g, cov_Ip_g, cv::noArray(), -1, stream);
  cv::cuda::multiply(mean_I_b, mean_p, cov_Ip_b, 1, -1, stream);
  cv::cuda::subtract(mean_Ip_b, cov_Ip_b, cov_Ip_b, cv::noArray(), -1, stream);

  cv::cuda::GpuMat prod1, prod2, prod3;
  cv::cuda::multiply(invrr, cov_Ip_r, prod1, 1, -1, stream);
  cv::cuda::multiply(invrg, cov_Ip_g, prod2, 1, -1, stream);
  cv::cuda::multiply(invrb, cov_Ip_b, prod3, 1, -1, stream);
  cv::cuda::add(prod1, prod2, prod2, cv::noArray(), -1, stream);
  cv::cuda::add(prod2, prod3, a_r, cv::noArray(), -1, stream);

  cv::cuda::multiply(invrg, cov_Ip_r, prod1, 1, -1, stream);
  cv::cuda::multiply(invgg, cov_Ip_g, prod2, 1, -1, stream);
  cv::cuda::multiply(invgb, cov_Ip_b, prod3, 1, -1, stream);
  cv::cuda::add(prod1, prod2, prod2, cv::noArray(), -1, stream);
  cv::cuda::add(prod2, prod3, a_g, cv::noArray(), -1, stream);

  cv::cuda::multiply(invrb, cov_Ip_r, prod1, 1, -1, stream);
  cv::cuda::multiply(invgb, cov_Ip_g, prod2, 1, -1, stream);
  cv::cuda::multiply(invbb, cov_Ip_b, prod3, 1, -1, stream);
  cv::cuda::add(prod1, prod2, prod2, cv::noArray(), -1, stream);
  cv::cuda::add(prod2, prod3, a_b, cv::noArray(), -1, stream);



  // Eqn. (15) in the paper
  cv::cuda::multiply(a_r, mean_I_r, prod1, 1, -1, stream);
  cv::cuda::multiply(a_g, mean_I_g, prod2, 1, -1, stream);
  cv::cuda::multiply(a_b, mean_I_b, prod3, 1, -1, stream);
  cv::cuda::subtract(mean_p, prod1, b, cv::noArray(), -1, stream);
  cv::cuda::subtract(b, prod2, b, cv::noArray(), -1, stream);
  cv::cuda::subtract(b, prod3, b, cv::noArray(), -1, stream);


  // Eqn. (16) in the paper;
  box_filter->apply(a_r, a_r, stream);
  box_filter->apply(a_g, a_g, stream);
  box_filter->apply(a_b, a_b, stream);
  box_filter->apply(b, b, stream);
  cv::cuda::multiply(a_r, Ichannels[0], a_r, 1, -1, stream);
  cv::cuda::multiply(a_g, Ichannels[1], a_g, 1, -1, stream);
  cv::cuda::multiply(a_b, Ichannels[2], a_b, 1, -1, stream);

  cv::cuda::add(a_r, a_g, a_r, cv::noArray(), -1, stream);
  cv::cuda::add(a_r, a_b, a_r, cv::noArray(), -1, stream);
  cv::cuda::add(a_r, b, b, cv::noArray(), -1, stream);

  return b;
}

GuidedFilter::GuidedFilter(const cv::cuda::GpuMat &I, int r, double eps) {
  CV_Assert(I.channels() == 1 || I.channels() == 3);

  if (I.channels() == 1)
    impl_ = new GuidedFilterMono(I, 2 * r + 1, eps);
  else
    impl_ = new GuidedFilterColor(I, 2 * r + 1, eps);
}

GuidedFilter::~GuidedFilter() {
  delete impl_;
}

cv::cuda::GpuMat GuidedFilter::filter(const cv::cuda::GpuMat &p,
                                      int depth, cv::cuda::Stream &stream) const {
  return impl_->filter(p, depth, stream);
}

cv::cuda::GpuMat guidedFilter(const cv::cuda::GpuMat &I,
                              const cv::cuda::GpuMat &p,
                              int r,
                              double eps,
                              int depth, cv::cuda::Stream &stream) {
  return GuidedFilter(I, r, eps).filter(p, depth, stream);
}
