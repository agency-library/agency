/*
 *  Copyright 2008-2017 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

//  This is the only Agency header that is guaranteed to 
//  change with every Agency release.
//
//  AGENCY_VERSION % 100 is the sub-minor version
//  AGENCY_VERSION / 100 % 1000 is the minor version
//  AGENCY_VERSION / 100000 is the major version

// XXX there are no leading zeros on AGENCY_VERSION because that is interpreted as an octal value
#define AGENCY_VERSION 300

#define AGENCY_MAJOR_VERSION     (AGENCY_VERSION / 100000)

#define AGENCY_MINOR_VERSION     (AGENCY_VERSION / 100 % 1000)

#define AGENCY_SUBMINOR_VERSION  (AGENCY_VERSION % 100)

