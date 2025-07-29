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
