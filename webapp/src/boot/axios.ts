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

/* eslint-disable @typescript-eslint/no-unused-expressions */
import { boot } from 'quasar/wrappers';
import type { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import axios, { AxiosError } from 'axios';
import type { DialogChainObject } from 'quasar';
import { Dialog, Loading } from 'quasar';
import { assertIsError } from 'src/helpers/utils';

declare module 'vue' {
  interface ComponentCustomProperties {
    $axios: AxiosInstance;
    $api: AxiosInstance;
  }
}

// Be careful when using SSR for cross-request state pollution
// due to creating a Singleton instance here;
// If any client changes this (global) instance, it might be a
// good idea to move this instance creation inside of the
// "export default () => {}" function below (which runs individually
// for each client)
const api = axios.create({ baseURL: '/' });

export default boot(({ app }) => {
  // for use inside Vue files (Options API) through this.$axios and this.$api

  app.config.globalProperties.$axios = axios;
  // ^ ^ ^ this will allow you to use this.$axios (for Vue Options API form)
  //       so you won't necessarily have to import axios in each vue file

  app.config.globalProperties.$api = api;
  // ^ ^ ^ this will allow you to use this.$api (for Vue Options API form)
  //       so you can easily perform requests against your app's API
});

export class ServerError extends Error {
  debugInfo?: string;
  type?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  error?: any;
}

/**
 * Add a 'api' prefix into an url.
 */
function getUrl(url: string) {
  return '/api/' + url;
}

function handleServerError(e: unknown) {
  if (e instanceof AxiosError) {
    if (e.response?.data) {
      const error = e.response.data.error;
      if (error) {
        const type = error.type;
        const ex = new ServerError(error?.message || e.response.data.error);
        console.error(error);
        ex.debugInfo = error.debugInfo;
        ex.type = type;
        ex.error = error;
        e = ex;
      }
    }
  }
  return e;
}

async function postApi<T>(
  url: string,
  params: unknown,
  loading?: () => void,
  options?: AxiosRequestConfig,
) {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let res: AxiosResponse<T, any>;
    if (options && options.method === 'DELETE') {
      res = await api.delete<T>(getUrl(url), { data: params });
    } else if (options && options.method === 'PUT') {
      res = await api.put<T>(getUrl(url), params, options);
    } else {
      res = await api.post<T>(getUrl(url), params, options);
    }
    loading && loading();
    return res;
  } catch (e: unknown) {
    loading && loading();
    throw handleServerError(e);
  }
}

async function postApiUi<T>(
  url: string,
  params: unknown,
  message: string,
  options?: AxiosRequestConfig,
) {
  const controller = new AbortController();
  let progressDlg: DialogChainObject | null = Dialog.create({
    message,
    progress: true, // we enable default settings
    persistent: true, // we want the user to not be able to close it
    ok: false, // we want the user to not be able to close it
    cancel: true,
    focus: 'none',
  });

  const startTime = Date.now();
  const intervalId = setInterval(() => {
    if (progressDlg) {
      const elapsedSeconds = Math.round((Date.now() - startTime) / 1000);
      progressDlg.update({
        message: `${message} (${elapsedSeconds}s)`,
      });
    }
  }, 1000);

  progressDlg.onCancel(() => {
    progressDlg = null;
    clearInterval(intervalId);
    controller.abort();
  });

  const loading = () => {
    if (progressDlg) {
      progressDlg.hide();
    }
    clearInterval(intervalId);
  };

  options = options || {};
  options.signal = controller.signal;
  try {
    return await postApi<T>(url, params, loading, options);
  } catch (e: unknown) {
    // The 'loading' function is called within postApi's catch block,
    // so the interval is cleared before the error is re-thrown.
    assertIsError(e);
    Dialog.create({
      title: 'Error',
      message: e.message,
    });
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function downloadFile(data: any, filename: string, mime: string, bom?: any) {
  const blobData = typeof bom !== 'undefined' ? [bom, data] : [data];
  const blob = new Blob(blobData, { type: mime || 'application/octet-stream' });
  const blobURL =
    window.URL && window.URL.createObjectURL
      ? window.URL.createObjectURL(blob)
      : window.webkitURL.createObjectURL(blob);
  const tempLink = document.createElement('a');
  tempLink.style.display = 'none';
  tempLink.href = blobURL;
  tempLink.setAttribute('download', filename);

  // Safari thinks _blank anchor are pop ups. We only want to set _blank
  // target if the browser does not support the HTML5 download attribute.
  // This allows you to download files in desktop safari if pop up blocking
  // is enabled.
  if (typeof tempLink.download === 'undefined') {
    tempLink.setAttribute('target', '_blank');
  }

  document.body.appendChild(tempLink);
  tempLink.click();

  // Fixes "webkit blob resource error 1"
  setTimeout(() => {
    document.body.removeChild(tempLink);
    window.URL.revokeObjectURL(blobURL);
  }, 200);
}

async function getFile(url: string, params?: unknown, loading?: () => void) {
  try {
    const res = await api.get(getUrl(url), { responseType: 'blob', params });
    loading && loading();
    downloadFile(res.data, res.headers['filename'] || 'google-ads.yaml', 'application/text');
    return res;
  } catch (e: unknown) {
    loading && loading();
    assertIsError(e);
    Dialog.create({
      title: 'Error',
      message: e.message,
    });
  }
}

async function getApi<T>(url: string, params?: unknown, loading?: () => void) {
  try {
    const res = await api.get<T>(getUrl(url), { params: params });
    loading && loading();
    return res;
  } catch (e: unknown) {
    loading && loading();
    throw handleServerError(e);
  }
}

async function getApiUi<T>(url: string, params: unknown, message: string) {
  Loading.show({ message });
  const loading = () => Loading.hide();
  try {
    return await getApi<T>(url, params, loading);
  } catch (e: unknown) {
    assertIsError(e);
    Dialog.create({
      title: 'Error',
      message: e.message,
    });
    // TODO: show e.debugInfo
  }
}

async function deleteApi<T>(url: string, loading?: () => void) {
  try {
    const res = await api.delete<T>(getUrl(url));
    loading && loading();
    return res;
  } catch (e: unknown) {
    loading && loading();
    throw handleServerError(e);
  }
}

async function deleteApiUi<T>(url: string, loadingMessage: string, confirmationMessage: string) {
  return new Promise((resolve) => {
    Dialog.create({
      title: 'Confirm',
      message: confirmationMessage,
      cancel: true,
      persistent: true,
    }).onOk(() => {
      Loading.show({ message: loadingMessage });
      const loading = () => Loading.hide();
      deleteApi<T>(url, loading)
        .then((response) => {
          resolve(response);
        })
        .catch((e: unknown) => {
          assertIsError(e);
          Dialog.create({
            title: 'Error',
            message: e.message,
          });
          resolve(undefined);
        });
    });
  });
}

export { api, postApi, postApiUi, getApi, getApiUi, getFile, deleteApiUi };
