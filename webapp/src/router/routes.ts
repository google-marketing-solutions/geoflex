/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import type { RouteRecordRaw } from 'vue-router';

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: () => import('layouts/MainLayout.vue'),
    children: [
      { path: '', component: () => import('pages/IndexPage.vue') },
      {
        path: 'configuration',
        component: () => import('pages/ConfigurationPage.vue'),
      },
      {
        path: 'datasources/:id?/:action?',
        component: () => import('pages/DataSourcesPage.vue'),
        props: true,
      },
      {
        path: 'experiments',
        component: () => import('pages/ExperimentsPage.vue'),
      },
      {
        path: 'designs',
        component: () => import('pages/DesignsPage.vue'),
      },
      {
        path: 'analysis/:designId?',
        name: 'analysis',
        component: () => import('pages/AnalysisPage.vue'),
      },
    ],
  },

  // Always leave this as last one,
  // but you can also remove it
  {
    path: '/:catchAll(.*)*',
    component: () => import('pages/ErrorNotFound.vue'),
  },
];

export default routes;
