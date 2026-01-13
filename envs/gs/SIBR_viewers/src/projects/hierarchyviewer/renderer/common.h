/*
 * Copyright (C) 2024, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef COMMON_H_INCLUDED
#define COMMON_H_INCLUDED

struct Point
{
	float xyz[3];
};

struct Point4
{
	float xyz[4];
};

struct Range
{
	int start;
	int end;
};

#endif