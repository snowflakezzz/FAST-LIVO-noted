#ifndef EARTH_LIB_H
#define EARTH_LIB_H

#include <Eigen/Geometry>

using Eigen::Vector3d;
using Eigen::Matrix3d;

const double WGS84_WIE = 7.2921151467E-5;       // 地球自转角速度
const double WGS84_F   = 0.0033528106647474805; // 扁率
const double WGS84_RA  = 6378137.0000000000;    // 长半轴a
const double WGS84_RB  = 6356752.3142451793;    // 短半轴b
const double WGS84_GM0 = 398600441800000.00;    // 地球引力常数
const double WGS84_E1  = 0.0066943799901413156; // 第一偏心率平方
const double WGS84_E2  = 0.0067394967422764341; // 第二偏心率平方

class Earth{
public:
    // compute norm of gravity by blh of anchor
    static double gravity(const Vector3d &blh){
        double sin2 = sin(blh[0]);
        sin2 *= sin2;

        return 9.7803267715 * (1 + 0.0052790414 * sin2 + 0.0000232718 * sin2 * sin2) +
               blh[2] * (0.0000000043977311 * sin2 - 0.0000030876910891) + 0.0000000000007211 * blh[2] * blh[2];
    }

    // 基准椭球的曲率半径
    static double RN(double lat) {
        double sinlat = sin(lat);
        return WGS84_RA / sqrt(1.0 - WGS84_E1 * sinlat * sinlat);
    }

    // coordinate from WGS84 to eccef
    static Vector3d blh2ecef(const Vector3d &blh){
        double coslat, sinlat, coslon, sinlon;
        double rnh, rn;

        coslat = cos(blh[0]);
        sinlat = sin(blh[0]);
        coslon = cos(blh[1]);
        sinlon = sin(blh[1]);

        rn = RN(blh[0]);
        rnh = rn + blh[2];

        return {rnh * coslat * coslon, rnh * coslat * sinlon, (rnh - rn * WGS84_E1) * sinlat};
    }

    // rotation from local to ecef
    static Matrix3d cne(const Vector3d &blh){
        double coslat, sinlat, coslon, sinlon;

        coslat = cos(blh[0]);
        sinlat = sin(blh[0]);
        coslon = cos(blh[1]);
        sinlon = sin(blh[1]);

        Matrix3d dcm;
        dcm(0, 0) = -sinlat * coslon;
        dcm(0, 1) = -sinlon;
        dcm(0, 2) = -coslat * coslon;

        dcm(1, 0) = -sinlat * sinlon;
        dcm(1, 1) = coslon;
        dcm(1, 2) = -coslat * sinlon;

        dcm(2, 0) = coslat;
        dcm(2, 1) = 0;
        dcm(2, 2) = -sinlat;

        return dcm;
    }
    // origin: 锚点, global: 待转换点blh坐标 输入必须为弧度
    static Vector3d global2local(const Vector3d &origin, const Vector3d &global){
        Vector3d ecef0  =   blh2ecef(origin);
        Matrix3d cn0e   =   cne(origin);

        Vector3d ecef1  =   blh2ecef(global);

        return cn0e.transpose() * (ecef1 - ecef0);
    }
};
#endif