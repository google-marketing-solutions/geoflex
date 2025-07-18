<template>
  <q-page padding>
    <div class="q-pa-md">
      <div class="text-h4 q-mb-md">Experiments</div>

      <!-- Main card structure with tabs -->
      <q-card class="my-card">
        <q-tabs
          v-model="activeTab"
          class="text-primary"
          active-color="primary"
          indicator-color="primary"
          align="justify"
        >
          <q-tab name="datasource" label="Data Source" />
          <q-tab name="constraints" label="Constraints & Parameters" />
          <q-tab
            name="designs"
            label="Test Designs"
            :disable="!selectedDataSource || !isExplored"
          />
        </q-tabs>

        <q-separator />

        <q-tab-panels v-model="activeTab" animated>
          <!-- Data Source Tab -->
          <q-tab-panel name="datasource">
            <div class="text-h6 q-mb-md">Select Data Source</div>

            <div class="row q-col-gutter-md">
              <div class="col-12">
                <q-card class="q-pa-md">
                  <div class="text-subtitle1 q-mb-sm">Available Data Sources</div>

                  <q-select
                    v-model="selectedDataSource"
                    :options="dataSourceOptions"
                    option-label="name"
                    option-value="id"
                    label="Choose a data source"
                    outlined
                    :loading="dataSourcesStore.loading"
                    @popup-show="loadDataSourcesOnOpen"
                    @update:model-value="handleDataSourceChange"
                  >
                    <template v-slot:option="scope">
                      <q-item v-bind="scope.itemProps">
                        <q-item-section>
                          <q-item-label>{{ scope.opt.name }}</q-item-label>
                          <q-item-label caption>{{
                            scope.opt.description || 'No description provided'
                          }}</q-item-label>
                        </q-item-section>
                        <q-item-section side>
                          <q-badge :color="scope.opt.data ? 'data' : 'yellow'"> </q-badge>
                        </q-item-section>
                      </q-item>
                    </template>
                    <template v-slot:no-option>
                      <q-item>
                        <q-item-section class="text-grey">
                          {{
                            dataSourcesStore.loading
                              ? 'Loading data sources...'
                              : 'No data sources found'
                          }}
                        </q-item-section>
                      </q-item>
                    </template>
                  </q-select>

                  <div class="text-caption q-mt-sm" v-if="!selectedDataSource">
                    Select a data source with geo units, date, and conversion metrics to continue
                  </div>
                </q-card>
              </div>
            </div>

            <q-card v-if="selectedDataSource && dataSourceLoaded" class="q-mt-lg q-pa-md">
              <div class="text-subtitle1 q-mb-sm">Data Source Preview</div>
              <div class="column q-gutter-y-md">
                <div>
                  <div class="text-weight-medium">Data Columns Configuration</div>
                  <div class="row q-col-gutter-md q-mt-sm">
                    <div class="col-12 col-md-3">
                      <q-input
                        v-model="selectedDataSource.columns.geoColumn"
                        label="Geo Unit Column"
                        outlined
                        dense
                        readonly
                      />
                    </div>
                    <div class="col-12 col-md-3">
                      <q-input
                        v-model="selectedDataSource.columns.dateColumn"
                        label="Date Column"
                        outlined
                        dense
                        readonly
                      />
                    </div>
                    <div class="col-12 col-md-3">
                      <q-list dense bordered :separator="true">
                        <q-item>
                          <q-item-section>
                            <q-item-label>Metric columns:</q-item-label>
                          </q-item-section>
                        </q-item>
                        <q-item v-for="key in selectedDataSource.columns.metricColumns" :key="key">
                          <q-item-section>
                            <q-item-label>{{ key }}</q-item-label>
                          </q-item-section>
                        </q-item>
                      </q-list>
                    </div>
                    <div class="col-12 col-md-3">
                      <q-input
                        v-model="selectedDataSource.columns.costColumn"
                        label="Cost Column"
                        outlined
                        dense
                        readonly
                        :disable="!selectedDataSource.columns.costColumn"
                      />
                    </div>
                  </div>
                </div>

                <q-separator />
                <div class="">
                  <div class="text-subtitle2 q-mb-sm">Primary Metric</div>
                  <metric-builder
                    v-model="selectedMetric"
                    :metric-columns="selectedDataSource.columns.metricColumns"
                    :cost-column="selectedDataSource.columns.costColumn"
                  />
                </div>

                <div>
                  <div class="text-subtitle2 q-mb-sm">Secondary Metrics</div>
                  <div v-for="(metric, index) in secondaryMetrics" :key="index" class="q-mb-md">
                    <div class="row items-center q-col-gutter-sm">
                      <div class="col">
                        <metric-builder
                          :model-value="metric"
                          @update:model-value="(val) => (secondaryMetrics[index] = val)"
                          :metric-columns="selectedDataSource.columns.metricColumns"
                          :cost-column="selectedDataSource.columns.costColumn"
                        />
                      </div>
                      <div class="col-auto">
                        <q-btn
                          flat
                          round
                          dense
                          icon="delete"
                          color="negative"
                          @click="removeSecondaryMetric(index)"
                        />
                      </div>
                    </div>
                  </div>
                  <q-btn
                    outline
                    color="primary"
                    icon="add"
                    label="Add Secondary Metric"
                    @click="addSecondaryMetric"
                  />
                </div>

                <q-separator />

                <div>
                  <div class="text-weight-medium">Data Sample</div>
                  <q-table
                    :rows="dataSample.rows"
                    :columns="dataSample.columns"
                    row-key="id"
                    dense
                    :pagination="{ rowsPerPage: 5 }"
                  />
                </div>

                <div v-if="selectedDataSource.data?.geoUnits?.length">
                  <div class="text-weight-medium">Data Overview</div>
                  <div class="row q-col-gutter-md">
                    <div class="col-12 col-md-4">
                      <q-card class="q-pa-sm">
                        <div class="text-subtitle2">Geo Units</div>
                        <div class="text-h5">{{ selectedDataSource.data.geoUnits.length }}</div>
                      </q-card>
                    </div>
                    <div class="col-12 col-md-4">
                      <q-card class="q-pa-sm">
                        <div class="text-subtitle2">Date Range</div>
                        <div class="text-h5">{{ datePeriod }}</div>
                      </q-card>
                    </div>
                    <div class="col-12 col-md-4">
                      <q-card class="q-pa-sm">
                        <div class="text-subtitle2">{{ getMetricName(selectedMetric) }}</div>
                        <div class="text-h5">{{ metricRange }}</div>
                      </q-card>
                    </div>
                  </div>
                </div>

                <div v-if="selectedDataSource.data?.geoUnits?.length" class="q-mt-md">
                  <q-separator class="q-my-md" />

                  <div class="text-weight-medium">Geo Units Assignment</div>
                  <div class="row items-center q-mb-md">
                    <div class="col">
                      <div class="text-body2">
                        Assign geo units to specific groups or exclude them from the test.
                        <span class="text-caption"
                          >Creating multiple test groups will automatically enable multi-cell
                          testing.</span
                        >
                      </div>
                    </div>
                    <div class="col-auto">
                      <q-input
                        v-model="geoSearch"
                        dense
                        outlined
                        placeholder="Search geo units"
                        class="q-mr-sm"
                        style="width: 200px"
                      >
                        <template v-slot:append>
                          <q-icon name="search" />
                        </template>
                      </q-input>
                    </div>
                  </div>

                  <q-table
                    :rows="filteredGeoUnits"
                    :columns="geoUnitsColumns"
                    row-key="geo"
                    dense
                    :pagination="{ rowsPerPage: 10 }"
                  >
                    <template v-slot:body-cell-assignment="props">
                      <q-td :props="props">
                        <q-select
                          v-model="props.row.assignment"
                          :options="computedAssignmentOptions"
                          dense
                          outlined
                          options-dense
                          style="min-width: 150px"
                          emit-value
                          map-options
                        >
                        </q-select>
                      </q-td>
                    </template>

                    <template v-slot:bottom="">
                      <div class="row full-width q-pa-sm">
                        <div class="col-auto q-mr-xl">
                          <q-btn
                            outline
                            color="primary"
                            icon="add_circle"
                            label="Add Test Group"
                            :disable="parameters.testGroups >= 10"
                            @click="addTestGroup"
                          />
                        </div>
                        <div
                          class="col-auto q-mr-md"
                          v-for="(count, group) in geoAssignmentCounts"
                          :key="group"
                        >
                          <q-chip
                            :color="
                              group === 'auto'
                                ? 'positive'
                                : group === 'control'
                                  ? 'blue'
                                  : group === 'exclude'
                                    ? 'negative'
                                    : 'orange'
                            "
                            text-color="white"
                            :icon="
                              group === 'control'
                                ? 'brightness_low'
                                : group === 'exclude'
                                  ? 'block'
                                  : 'science'
                            "
                          >
                            {{ formatGroupLabel(group) }}: {{ count }}
                          </q-chip>
                        </div>
                      </div>
                    </template>
                  </q-table>
                </div>
              </div>
            </q-card>
          </q-tab-panel>

          <!-- Constraints & Parameters Tab -->
          <q-tab-panel name="constraints">
            <div class="text-h6 q-mb-md">Constraints & Parameters</div>

            <q-card class="q-pa-md">
              <div class="text-subtitle1">Test Parameters and Exploration Axes</div>
              <q-separator class="q-my-md" />

              <div class="row q-mb-md">
                <div class="col">
                  <q-banner rounded class="q-mb-md">
                    <template v-slot:avatar>
                      <q-icon name="info" color="primary" />
                    </template>
                    <div class="text-body2">
                      <strong>Single value parameters</strong> act as constraints.
                      <strong>Multiple value parameters</strong> become exploration axes, generating
                      designs for all combinations. The tool will find the most powerful
                      experimental designs based on your data and parameters.
                    </div>
                  </q-banner>
                </div>
              </div>

              <div class="row q-col-gutter-md">
                <!-- Left Column -->
                <div class="col-12 col-md-6">
                  <div class="text-subtitle2 q-mb-sm">Core Parameters</div>

                  <div class="q-gutter-y-md">
                    <!-- Methodology -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Test Methodology</div>
                        </div>
                        <div class="col-auto">
                          <q-badge
                            v-if="
                              parameters.methodology.length > 1 ||
                              parameters.methodology.length === 0
                            "
                            color="purple"
                            label="Exploration Axis"
                          />
                          <q-badge v-else color="blue" label="Constraint" />
                        </div>
                      </div>
                      <q-option-group
                        v-model="parameters.methodology"
                        :options="methodologyOptionsDetailed"
                        type="checkbox"
                        dense
                      />
                      <div class="text-caption q-pl-md q-pt-xs">
                        <q-icon name="info" size="xs" color="grey" /> Select multiple to explore all
                        combinations
                      </div>
                    </div>

                    <!-- Structure -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Test Structure</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="blue" label="Constraint" />
                        </div>
                      </div>
                      <q-option-group
                        v-model="parameters.structure"
                        :options="structureOptionsDetailed"
                        type="radio"
                        dense
                      />
                      <div v-if="parameters.structure === 'multi-cell'" class="q-mt-sm">
                        <q-input
                          v-model.number="parameters.testGroups"
                          type="number"
                          label="Number of test groups"
                          outlined
                          dense
                          min="2"
                          max="5"
                          hint="How many test variations to compare (excluding control)"
                        />
                      </div>
                    </div>

                    <!-- Target power -->
                    <div class="row items-center q-mt-sm">
                      <div class="col-6">
                        <q-input
                          v-model.number="parameters.power"
                          type="number"
                          outlined
                          dense
                          min="50"
                          max="99"
                          step="1"
                          label="Target Power (%)"
                        >
                          <template v-slot:append>
                            <q-icon name="percent" />
                          </template>
                          <template v-slot:hint>
                            <span
                              >Probability of detecting the specified effect (typically
                              80-90%)</span
                            >
                          </template>
                        </q-input>
                      </div>
                    </div>

                    <!-- Significance Level -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Significance Level (α)</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="blue" label="Constraint" />
                        </div>
                      </div>
                      <div class="row items-center">
                        <div class="col-6">
                          <q-input
                            v-model.number="parameters.alpha"
                            type="number"
                            outlined
                            dense
                            min="0.01"
                            max="0.2"
                            step="0.01"
                          >
                            <template v-slot:hint>
                              <span
                                >Threshold for statistical significance (typically 0.05 or
                                0.1)</span
                              >
                            </template>
                          </q-input>
                        </div>
                      </div>
                    </div>

                    <!-- Budget -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">
                            Budget
                            <q-icon
                              name="info_outline"
                              size="xs"
                              class="q-ml-xs"
                              v-if="isBudgetSectionDisabled"
                            >
                              <q-tooltip
                                >Budget controls are disabled because the selected data source does
                                not have a cost column defined.</q-tooltip
                              >
                            </q-icon>
                          </div>
                        </div>
                      </div>
                      <div class="row q-col-gutter-sm items-start">
                        <div class="col-8">
                          <q-select
                            v-model="parameters.budget"
                            label="Budget Value(s) per Test Group"
                            outlined
                            dense
                            multiple
                            use-chips
                            use-input
                            new-value-mode="add-unique"
                            hide-dropdown-icon
                            input-debounce="0"
                            :rules="[budgetValidationRule]"
                            :disable="isBudgetSectionDisabled"
                            hint="Enter numeric values. Press Enter or Tab to add. Leave empty for A/B test (0 budget)."
                          />
                        </div>
                        <div class="col-4">
                          <q-select
                            v-model="parameters.budgetType"
                            :options="filteredBudgetTypeOptions"
                            label="Type"
                            outlined
                            dense
                            emit-value
                            map-options
                            option-value="value"
                            option-label="label"
                            :disable="isBudgetSectionDisabled"
                          >
                            <template v-slot:selected-item="scope">
                              <q-item-label :title="scope.opt.desc">
                                {{ scope.opt.label }}</q-item-label
                              >
                            </template>
                            <template v-slot:option="scope">
                              <q-item v-bind="scope.itemProps" :title="scope.opt.desc">
                                <q-item-section>
                                  <q-item-label>{{ scope.opt.label }}</q-item-label>
                                  <q-item-label caption>{{ scope.opt.desc }}</q-item-label>
                                </q-item-section>
                              </q-item>
                            </template>
                            <template v-slot:no-option>
                              <q-item>
                                <q-item-section class="text-grey">
                                  Percentage budget type not available if cost data is missing or
                                  all zeros.
                                </q-item-section>
                              </q-item>
                            </template>
                          </q-select>
                        </div>
                      </div>
                      <div
                        class="text-caption q-pl-sm q-pt-xs"
                        v-if="parameters.structure === 'multi-cell'"
                      >
                        Number of budget values must match the number of test groups ({{
                          parameters.testGroups
                        }}) or be empty.
                      </div>
                      <div class="text-caption q-pl-sm q-pt-xs" v-else>
                        Budget should be a single value or empty for a single-cell test.
                      </div>
                    </div>
                  </div>
                </div>

                <!-- Right Column -->
                <div class="col-12 col-md-6">
                  <div class="text-subtitle2 q-mb-sm">Advanced Parameters</div>

                  <div class="q-gutter-y-md">
                    <!-- Hypothesis Type -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Hypothesis Type</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="blue" label="Constraint" />
                        </div>
                      </div>
                      <q-option-group
                        v-model="parameters.hypothesisType"
                        :options="hypothesisOptionsDetailed"
                        type="radio"
                        dense
                      />
                    </div>

                    <!-- Test Duration Range -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Test Duration Range (weeks)</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="purple" label="Exploration Axis" />
                        </div>
                      </div>
                      <div class="row q-col-gutter-sm">
                        <div class="col-6">
                          <q-input
                            v-model.number="parameters.durationMin"
                            type="number"
                            outlined
                            dense
                            min="1"
                            step="1"
                            label="Minimum"
                          >
                            <template v-slot:append>
                              <span class="text-caption">wks</span>
                            </template>
                          </q-input>
                        </div>
                        <div class="col-6">
                          <q-input
                            v-model.number="parameters.durationMax"
                            type="number"
                            outlined
                            dense
                            min="1"
                            step="1"
                            label="Maximum"
                          >
                            <template v-slot:append>
                              <span class="text-caption">wks</span>
                            </template>
                          </q-input>
                        </div>
                      </div>
                      <div class="text-caption q-pl-md q-pt-xs">
                        <q-icon name="info" size="xs" color="grey" /> The tool will explore test
                        durations between the minimum and maximum values
                      </div>
                    </div>

                    <!-- Effect Scope -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Effect Scope</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="blue" label="Constraint" />
                        </div>
                      </div>
                      <q-option-group
                        v-model="parameters.effectScope"
                        :options="effectScopeOptions"
                        type="radio"
                        dense
                      />
                    </div>

                    <!-- Cell Volume Constraints -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Cell Volume Constraints</div>
                        </div>
                      </div>
                      <q-card flat bordered class="q-pa-md">
                        <q-toggle
                          v-model="parameters.cellVolumeConstraint.enabled"
                          label="Enable Constraints"
                        />

                        <div
                          v-if="parameters.cellVolumeConstraint.enabled"
                          class="q-gutter-y-md q-mt-sm"
                        >
                          <q-select
                            v-model="parameters.cellVolumeConstraint.type"
                            :options="cellVolumeConstraintTypeOptions"
                            label="Constraint Type"
                            outlined
                            dense
                            emit-value
                            map-options
                          />

                          <q-select
                            v-if="
                              parameters.cellVolumeConstraint.type === 'max_percentage_of_metric'
                            "
                            v-model="parameters.cellVolumeConstraint.metric_column"
                            :options="selectedDataSource.columns.metricColumns"
                            label="Metric Column"
                            outlined
                            dense
                            hint="Select metric to use for the constraint"
                          />

                          <div class="text-subtitle2 q-mt-md">Constraint Values per Cell</div>
                          <div
                            v-for="(label, index) in cellLabels"
                            :key="index"
                            class="row items-center q-col-gutter-sm"
                          >
                            <div class="col-4">
                              <span class="text-body2">{{ label }}</span>
                            </div>
                            <div class="col-8">
                              <q-input
                                v-model.number="parameters.cellVolumeConstraint.values[index]"
                                type="number"
                                :label="`Max value for ${label}`"
                                outlined
                                dense
                                clearable
                                :hint="
                                  parameters.cellVolumeConstraint.type ===
                                  'max_percentage_of_metric'
                                    ? 'As a percentage (e.g., 0.1 for 10%)'
                                    : 'As a number of geos'
                                "
                              />
                            </div>
                          </div>
                        </div>
                      </q-card>
                    </div>

                    <!-- Simulation Per Trial  -->
                    <div>
                      <q-card>
                        <q-card-section class="q-py-sm1">
                          <div class="text-subtitle2">Exploration advanced parameters</div>
                        </q-card-section>
                        <q-card-section class="">
                          <div class="row q-my-sm">
                            <div class="col-6">
                              <q-input
                                v-model.number="parameters.maxTrials"
                                label="Max Trials"
                                type="number"
                                outlined
                                dense
                                min="1"
                                step="1"
                                hint="The maximum number of valid trials to run (leave empty for default)"
                              />
                            </div>
                          </div>
                          <div class="row q-my-sm">
                            <div class="col-6">
                              <q-input
                                v-model.number="parameters.nDesigns"
                                label="# of designs"
                                type="number"
                                outlined
                                dense
                                min="1"
                                step="1"
                                hint="The number of designs to explore (by default 5)"
                              />
                            </div>
                          </div>
                        </q-card-section>
                      </q-card>
                    </div>

                    <!-- Budget Allocation
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Budget Allocation Method</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="blue" label="Constraint" />
                        </div>
                      </div>
                      <q-select
                        v-model="parameters.budgetAllocation"
                        :options="budgetOptionsDetailed"
                        outlined
                        dense
                        option-label="label"
                        option-value="value"
                        emit-value
                        map-options
                        hint="How budget is distributed across test groups"
                      />
                    </div>
                  --></div>
                </div>
              </div>

              <q-separator class="q-my-md" />

              <div class="row items-center">
                <div class="col">
                  <q-banner
                    v-if="explorationValidationIssues.length > 0"
                    inline-actions
                    dense
                    class="text-white bg-red q-mb-md"
                  >
                    <template v-slot:avatar>
                      <q-icon name="warning" color="white" />
                    </template>
                    <b>Please address the following issues:</b>
                    <ul>
                      <li v-for="issue in explorationValidationIssues" :key="issue">{{ issue }}</li>
                    </ul>
                  </q-banner>
                </div>
              </div>
              <div class="row items-center justify-end">
                <div class="col-auto">
                  <q-btn label="Reset" color="grey" flat class="q-mr-sm" @click="resetParameters" />
                  <q-btn
                    label="Run Exploration"
                    color="primary"
                    :disable="
                      !selectedDataSource ||
                      !dataSourceLoaded ||
                      explorationValidationIssues.length > 0
                    "
                    @click="runExploration"
                  />
                </div>
              </div>
            </q-card>
          </q-tab-panel>

          <!-- Test Designs Tab -->
          <q-tab-panel name="designs">
            <div class="text-h6 q-mb-md">Test Designs</div>

            <q-card class="q-pa-md">
              <div class="row items-center q-mb-md">
                <div class="col">
                  <div class="text-subtitle1">Found {{ testDesigns.length }} designs</div>
                </div>
                <div class="col-auto">
                  <q-select
                    v-model="sortBy"
                    :options="sortOptions"
                    label="Sort by"
                    outlined
                    dense
                    style="min-width: 200px"
                  />
                </div>
              </div>

              <div v-for="(design, index) in sortedDesigns" :key="index" class="q-mb-lg">
                <q-card bordered>
                  <q-card-section>
                    <div class="row items-center">
                      <div class="col">
                        <div class="text-h6">Design #{{ index + 1 }}</div>
                        <div class="text-caption">Methodology: {{ design.methodology }}</div>
                        <div class="text-caption">Duration: {{ design.runtime_weeks }} weeks</div>
                      </div>
                      <div class="col-auto">
                        <q-btn-group flat>
                          <q-btn color="primary" icon="visibility" @click="viewDesign(design)" />
                          <q-btn color="positive" icon="download" @click="downloadDesign(design)" />
                        </q-btn-group>
                      </div>
                    </div>
                  </q-card-section>

                  <q-separator />

                  <q-card-section>
                    <div class="row q-col-gutter-md">
                      <!-- Parameters Summary -->
                      <!-- <div class="col-12 col-md-6">
                        <div class="text-subtitle2">Parameters</div>
                        <q-list dense>
                          <q-item v-for="(value, key) in design.parameters" :key="key">
                            <q-item-section>
                              <q-item-label caption>{{ formatKey(key) }}</q-item-label>
                              <q-item-label>{{ value }}</q-item-label>
                            </q-item-section>
                          </q-item>
                        </q-list>
                      </div> -->

                      <!-- Statistical Properties -->
                      <div class="col-12 col-md-6">
                        <div class="text-subtitle2">Statistical Properties</div>
                        <div class="row q-col-gutter-md">
                          <div class="col-6">
                            <q-item dense>
                              <q-item-section>
                                <q-item-label caption
                                  >MDE ({{
                                    getMetricName(design.parameters.primary_metric) ||
                                    'Primary Metric'
                                  }})</q-item-label
                                >
                                <q-item-label class="text-primary text-weight-bold">{{
                                  design.mde ? design.mde.toFixed(2) + '%' : 'N/A'
                                }}</q-item-label>
                              </q-item-section>
                            </q-item>
                          </div>
                        </div>
                      </div>
                    </div>

                    <!-- Groups Summary -->
                    <div class="q-mt-md">
                      <div class="text-subtitle2">Groups</div>
                      <div class="row q-col-gutter-md">
                        <div
                          v-for="(geos, groupName) in design.groups"
                          :key="groupName as string"
                          class="col-12 col-md-4"
                        >
                          <q-card flat bordered>
                            <q-card-section class="q-py-sm bg-primary text-white row items-center">
                              <div class="text-subtitle2 col">
                                {{ groupName }} ({{ geos.length }} geos)
                              </div>
                              <div class="col-auto">
                                <q-btn
                                  flat
                                  dense
                                  round
                                  icon="file_download"
                                  color="white"
                                  @click="exportDesignGroupToCsv(design, groupName as string)"
                                >
                                  <q-tooltip>Export {{ groupName }} geos as CSV</q-tooltip>
                                </q-btn>
                              </div>
                            </q-card-section>
                            <q-card-section class="q-pa-sm">
                              <div
                                class="geo-chips-container"
                                style="max-height: 200px; overflow-y: auto"
                              >
                                <q-chip
                                  v-for="geo in geos"
                                  :key="geo"
                                  :color="
                                    isFixedGeo(geo, groupName as string) ? 'orange' : 'primary'
                                  "
                                  text-color="white"
                                  size="sm"
                                  class="q-ma-xs"
                                >
                                  {{ geo }}
                                </q-chip>
                              </div>
                            </q-card-section>
                          </q-card>
                        </div>
                      </div>
                    </div>
                  </q-card-section>

                  <!-- Expandable section with charts -->
                  <!--
                  <q-expansion-item
                    icon="analytics"
                    label="View metrics"
                    caption="Pre-test metrics comparison"
                  >
                    <q-card-section>
                      <div class="row q-col-gutter-md">
                        <div class="col-12 col-md-6">
                          !-- Placeholder for chart component --
                          <div class="bg-grey-3 flex flex-center" style="height: 200px">
                            Group Balance Chart
                          </div>
                        </div>
                        <div class="col-12 col-md-6">
                          !-- Placeholder for chart component --
                          <div class="bg-grey-3 flex flex-center" style="height: 200px">
                            Time Series Comparison
                          </div>
                        </div>
                      </div>
                    </q-card-section>
                  </q-expansion-item>
                  -->
                </q-card>
              </div>
            </q-card>
          </q-tab-panel>
        </q-tab-panels>
      </q-card>
    </div>

    <!-- Design Detail Dialog -->
    <q-dialog v-model="designDetailDialog" maximized persistent>
      <q-card>
        <q-card-section class="row items-center">
          <div class="text-h6">Design Details</div>
          <q-space />
          <q-btn icon="close" flat round dense v-close-popup />
        </q-card-section>

        <q-separator />

        <q-card-section v-if="selectedDesign" class="q-pa-md">
          <div class="row q-col-gutter-md">
            <!-- Design metadata -->
            <div class="col-12 col-md-4">
              <q-card class="q-pa-md">
                <div class="text-subtitle1">Design Summary</div>
                <q-list dense>
                  <q-item v-for="(value, key) in selectedDesign.parameters" :key="key">
                    <q-item-section>
                      <q-item-label caption>{{ formatKey(key) }}</q-item-label>
                      <q-item-label
                        v-if="typeof value === 'object' && value !== null"
                        class="text-caption"
                      >
                        <pre>{{ JSON.stringify(value, null, 2) }}</pre>
                      </q-item-label>
                      <q-item-label v-else>{{ value }}</q-item-label>
                    </q-item-section>
                  </q-item>
                </q-list>
              </q-card>
            </div>

            <!-- Statistics -->
            <div class="col-12 col-md-8">
              <q-card class="q-pa-md">
                <div class="text-subtitle1">Statistical Properties</div>
                <div class="row q-col-gutter-md">
                  <div class="col-6 col-md-3">
                    <div class="text-caption">
                      MDE ({{
                        getMetricName(selectedDesign.parameters.primary_metric) || 'Primary Metric'
                      }})
                    </div>
                    <div class="text-h5">
                      {{ selectedDesign.mde ? selectedDesign.mde.toFixed(1) + '%' : 'N/A' }}
                    </div>
                  </div>
                  <div class="col-6 col-md-4">
                    <div class="text-caption">Duration</div>
                    <div class="text-h5">{{ selectedDesign.runtime_weeks }} wks</div>
                  </div>
                </div>
                <!-- <q-list dense>
                  <q-item v-for="(value, key) in selectedDesign.evaluation_results?.all_metric_results_per_cell" :key="key">
                    <q-item-section>
                      <q-item-label caption>{{ formatKey(key) }}</q-item-label>
                      <q-item-label
                        v-if="typeof value === 'object' && value !== null"
                        class="text-caption"
                      >
                        <pre>{{ JSON.stringify(value, null, 2) }}</pre>
                      </q-item-label>
                      <q-item-label v-else>{{ value }}</q-item-label>
                    </q-item-section>
                  </q-item>
                </q-list> -->
              </q-card>

              <!-- Power charts -->
              <q-card class="q-pa-md q-mt-md">
                <div class="text-subtitle1">Power Analysis</div>
                <div class="bg-grey-3 flex flex-center q-mt-sm" style="height: 200px">
                  Power Analysis Chart
                </div>
              </q-card>
            </div>

            <!-- Group details -->
            <div class="col-12">
              <q-tabs
                v-model="groupTab"
                class="text-primary"
                active-color="primary"
                indicator-color="primary"
                align="justify"
              >
                <q-tab
                  v-for="(group, groupName) in selectedDesign.groups"
                  :key="groupName"
                  :name="groupName"
                  :label="groupName"
                />
              </q-tabs>

              <q-separator />

              <q-tab-panels v-model="groupTab" animated>
                <q-tab-panel
                  v-for="(group, groupName) in selectedDesign.groups"
                  :key="groupName"
                  :name="groupName"
                >
                  <div class="row q-col-gutter-md">
                    <div class="col-12 col-md-4">
                      <q-card class="q-pa-md">
                        <div class="text-subtitle1">{{ groupName }} Summary</div>
                        <q-list dense>
                          <q-item>
                            <q-item-section>
                              <q-item-label caption>Geo Count</q-item-label>
                              <q-item-label>{{ group.length }}</q-item-label>
                            </q-item-section>
                          </q-item>
                        </q-list>
                      </q-card>
                    </div>

                    <div class="col-12 col-md-8">
                      <q-card class="q-pa-md">
                        <div class="text-subtitle1">Time Series</div>
                        <div class="bg-grey-3 flex flex-center q-mt-sm" style="height: 200px">
                          Time Series Chart
                        </div>
                      </q-card>
                    </div>
                  </div>

                  <q-card class="q-pa-md q-mt-md">
                    <div class="text-subtitle1">Geo Units</div>
                    <q-table
                      :rows="group"
                      :columns="geoDetailColumns"
                      row-key="geo"
                      dense
                      :pagination="{ rowsPerPage: 10 }"
                    />
                  </q-card>
                </q-tab-panel>
              </q-tab-panels>
            </div>
          </div>
        </q-card-section>

        <q-card-actions align="right">
          <q-btn color="primary" label="Download" @click="downloadDesign(selectedDesign)" />
          <q-btn color="grey" label="Close" v-close-popup />
        </q-card-actions>
      </q-card>
    </q-dialog>
  </q-page>
</template>

<script setup lang="ts">
import { ref, computed, reactive, watch } from 'vue';
import { useDataSourcesStore, type DataSource } from 'src/stores/datasources';
import type { QTableColumn } from 'quasar';
import { useQuasar, exportFile } from 'quasar';
import { postApiUi } from 'src/boot/axios';
import MetricBuilder from 'src/components/MetricBuilder.vue';
import type { AnyMetric, ExperimentDesign, ExperimentExploreResponse } from 'src/components/models';

const dataSourcesStore = useDataSourcesStore();
const $q = useQuasar();

// Tab state
const activeTab = ref('datasource');

// Data source state
const selectedDataSource = ref<DataSource | null>(null);
const dataSourceLoaded = ref(false);
const selectedMetric = ref<AnyMetric>('');
const secondaryMetrics = ref<AnyMetric[]>([]);

// Derived data source properties
const dataSourceOptions = computed(() => dataSourcesStore.datasources);

// Function to load data sources when dropdown is opened
async function loadDataSourcesOnOpen() {
  if (!dataSourcesStore.isLoaded && !dataSourcesStore.loading) {
    await dataSourcesStore.loadDataSources();
  }
}

// Data source handling
const handleDataSourceChange = async (dataSource) => {
  if (!dataSource) {
    dataSourceLoaded.value = false;
    return;
  }

  try {
    if (!dataSource.data) {
      await dataSourcesStore.loadDataSourceData(dataSource);
    }

    dataSourceLoaded.value = true;

    // Set default metric
    if (dataSource.columns.metricColumns.length > 0) {
      selectedMetric.value = dataSource.columns.metricColumns[0];
    }
    secondaryMetrics.value = []; // Reset secondary metrics

    // Populate geo units
    if (dataSource.data && dataSource.data.geoUnits) {
      populateGeoUnits(dataSource);
    }
  } catch (error) {
    console.error('Error loading data source:', error);
  }
};

function populateGeoUnits(dataSource) {
  const { geoColumn } = dataSource.columns;
  const geoGroups = {};
  dataSource.data.rawRows.forEach((row) => {
    const geo = row[geoColumn];
    geoGroups[geo] = {};
  });
  geoUnits.value = Object.keys(geoGroups).map((geo) => {
    return {
      geo,
      assignment: 'auto',
    };
  });
}

const datePeriod = computed(() => {
  if (!selectedDataSource.value?.data?.uniqueDates?.length) return 'No dates';
  const dates = selectedDataSource.value.data.uniqueDates;
  return `${dates[0]} - ${dates[dates.length - 1]}`;
});

const metricRange = computed(() => {
  if (!selectedDataSource.value?.data?.metrics || !selectedMetric.value) return 'N/A';

  const metricName = getMetricName(selectedMetric.value);
  if (!metricName) return 'N/A';

  const metric = selectedDataSource.value.data.metrics.find((m) => m.name === metricName);
  if (!metric) return 'N/A';

  return `${formatNumber(metric.min)} - ${formatNumber(metric.max)}`;
});

// Data sample for preview
const dataSample = computed(() => {
  if (!selectedDataSource.value || !selectedDataSource.value.data?.rawRows) {
    return {
      columns: [] as QTableColumn[],
      rows: [],
    };
  }

  const rawRows = selectedDataSource.value.data.rawRows;
  const sampleRows = rawRows.slice(0, 100).map((row, index) => ({ id: index, ...row }));

  // Extract column definitions from the first row
  const columns = Object.keys(rawRows[0] || {}).map((key) => ({
    name: key,
    label: key,
    field: key,
    align: typeof rawRows[0][key] === 'number' ? 'right' : 'left',
  }));

  return {
    columns: columns as QTableColumn[],
    rows: sampleRows,
  };
});

const geoSearch = ref('');
const geoUnits = ref([]);

// Geo units table
const geoUnitsColumns: QTableColumn[] = [
  { name: 'geo', label: 'Geo Unit', field: 'geo', align: 'left' },
  { name: 'assignment', label: 'Assignment', field: 'assignment', align: 'center' },
];

// Computed properties for geo assignment
const computedAssignmentOptions = computed(() => {
  const options = [
    { label: 'Auto', value: 'auto' },
    { label: 'Control Group', value: 'control' },
    { label: 'Test Group', value: 'test' },
  ];

  // Add additional test groups if needed
  for (let i = 1; i < parameters.testGroups; i++) {
    options.push({ label: `Test Group ${String.fromCharCode(65 + i)}`, value: `test-${i + 1}` });
  }

  options.push({ label: 'Exclude', value: 'exclude' });

  return options;
});

const filteredGeoUnits = computed(() => {
  if (!geoSearch.value) return geoUnits.value;

  const query = geoSearch.value.toLowerCase();
  return geoUnits.value.filter((unit) => unit.geo.toLowerCase().includes(query));
});

// return counts of each group to show badges
const geoAssignmentCounts = computed(() => {
  const counts = {
    auto: 0,
    control: 0,
    test: 0,
    exclude: 0,
  };

  // Initialize additional test groups
  for (let i = 1; i < parameters.testGroups; i++) {
    counts[`test-${i + 1}`] = 0;
  }

  // Count assignments
  geoUnits.value.forEach((unit) => {
    if (counts[unit.assignment] !== undefined) {
      counts[unit.assignment]++;
    } else {
      counts.auto++;
    }
  });

  return counts;
});

function addTestGroup() {
  parameters.structure = 'multi-cell';
  parameters.testGroups++;
}

function formatGroupLabel(group) {
  if (group === 'auto') return 'Auto';
  if (group === 'control') return 'Control';
  if (group === 'test') return 'Test';
  if (group === 'exclude') return 'Excluded';
  if (group.startsWith('test-')) {
    const groupNum = parseInt(group.split('-')[1]);
    return `Test ${String.fromCharCode(64 + groupNum)}`;
  }
  return group;
}

const methodologyOptionsDetailed = [
  // {
  //   label: 'Time Based Regression with Matched Markets (TBR-MM)',
  //   value: 'TBR-MM',
  //   desc: 'Regression-based methodology that uses time series data with matched markets',
  // },
  // {
  //   label: 'Time Based Regression with Random Assignment (TBR)',
  //   value: 'TBR',
  //   desc: 'Regression-based methodology that uses time series data with random geo assignment',
  // },
  // {
  //   label: 'Trimmed Match (TM)',
  //   value: 'TM',
  //   desc: 'Matches geo units based on similarity and trims outliers',
  // },
  {
    label: 'Geo Based Regression (GBR)',
    value: 'GBR',
    desc: 'The original and simplest geo experiment methodology',
  },
  {
    label: 'Synthetic Control',
    value: 'SyntheticControls',
    desc: '',
  },
];

const structureOptionsDetailed = [
  {
    label: 'Single-cell (A/B)',
    value: 'single-cell',
    desc: 'Standard A/B test with one control and one test group',
  },
  {
    label: 'Multi-cell (A/B/C...)',
    value: 'multi-cell',
    desc: 'Multiple test groups each receiving different variations',
  },
];

const hypothesisOptionsDetailed = [
  {
    label: 'One-sided (recommended)',
    value: 'one-sided',
    desc: 'Examines effect in one direction only (e.g., whether treatment performs better than control)',
  },
  {
    label: 'Two-sided',
    value: 'two-sided',
    desc: 'Examines effect in both directions (better or worse)',
  },
];

const effectScopeOptions = [
  {
    label: 'All Geos',
    value: 'all_geos',
    desc: 'The goal of the experiment is to estimate the average effect of the treatment (ads) over all geos',
  },
  {
    label: 'Treatment Geos',
    value: 'treatment_geos',
    desc: 'The goal of the experiment is to estimate the average effect of the treatment (ads) over the geos in the treatment group',
  },
];

/*
const budgetOptionsDetailed = [
  {
    label: 'Equal across regions',
    value: 'equal',
    desc: 'Same budget allocated to each geo unit regardless of size',
  },
  {
    label: 'Proportional to market size',
    value: 'proportional',
    desc: 'Budget allocated in proportion to the size or importance of each geo',
  },
  {
    label: 'Optimized based on historical data',
    value: 'optimized',
    desc: 'Budget allocated based on historical performance data',
  },
];
*/

const getDefaultParameters = () => ({
  // Core parameters
  methodology: [] as string[], // Empty array means "explore all methodologies"
  structure: 'single-cell',
  testGroups: 1, // Only used when structure is 'multi-cell'

  power: 80, // Target power

  // Duration settings
  durationMin: 4,
  durationMax: 8,

  alpha: 0.05,
  hypothesisType: 'one-sided',
  effectScope: 'all_geos',

  // Budget parameters
  budget: [] as string[], // Array of budget values (as strings from input)
  budgetType: 'percentage_change', // 'percentage_change', 'daily_budget', 'total_budget'

  // Cell volume constraints
  cellVolumeConstraint: {
    enabled: false,
    type: 'max_geos',
    values: [null, null],
    metric_column: null,
  },

  simulationsPerTrial: undefined,
  maxTrials: undefined,
  nDesigns: undefined,
});

// Initialize parameters with default values
const parameters = reactive(getDefaultParameters());

// Watch for changes in testGroups count
watch(
  () => parameters.testGroups,
  (newCount, oldCount) => {
    // If we reduced the number of test groups
    if (newCount < oldCount) {
      // Find geo units assigned to test groups that no longer exist
      geoUnits.value.forEach((unit) => {
        // Check if this is a test group assignment (test-2, test-3, etc)
        if (unit.assignment.startsWith('test-')) {
          // Extract the group number (test-2 -> 2)
          const groupNum = parseInt(unit.assignment.split('-')[1]);

          // If this group number is higher than our current count, reset to auto
          // +1 because test groups are 1-indexed
          if (groupNum > newCount) {
            unit.assignment = 'auto';
            console.log(`Reset ${unit.geo} from ${unit.assignment} to auto`);
          }
        }
      });
    }
  },
);

// Watch for changes in structure type
watch(
  () => parameters.structure,
  (newStructure) => {
    if (newStructure === 'single-cell') {
      // When switching to single-cell, set testGroups to 1
      parameters.testGroups = 1;

      // Reset any geo units assigned to test-2, test-3, etc. to 'test'
      geoUnits.value.forEach((unit) => {
        if (unit.assignment.startsWith('test-') && unit.assignment !== 'test') {
          // Convert any test-2, test-3, etc. to just 'test'
          unit.assignment = 'test';
        }
      });
    } else if (newStructure === 'multi-cell' && parameters.testGroups < 2) {
      // When switching to multi-cell, ensure we have at least 2 test groups
      parameters.testGroups = 2;
    }
  },
);

const resetParameters = () => {
  // Reset parameters to default values
  Object.assign(parameters, getDefaultParameters());
};

const sortOptions = [
  { label: 'Lowest MDE', value: 'mde' },
  { label: 'Shortest Duration', value: 'duration' },
];

const cellVolumeConstraintTypeOptions = [
  { label: 'Max Geos per Cell', value: 'max_geos' },
  { label: 'Max % of Metric per Cell', value: 'max_percentage_of_metric' },
];

// --- Computed properties for UI logic and validation (Order Matters) ---

const nCells = computed(() => {
  return parameters.structure === 'multi-cell' ? parameters.testGroups + 1 : 2;
});

const cellLabels = computed(() => {
  const labels = ['Control'];
  if (parameters.structure === 'single-cell') {
    labels.push('Test');
  } else {
    // For multi-cell
    for (let i = 0; i < parameters.testGroups; i++) {
      labels.push(`Test Group ${i + 1}`);
    }
  }
  return labels;
});

const fullWeekCount = computed(() => {
  if (!selectedDataSource.value?.data?.uniqueDates?.length) return 0;
  return Math.floor(selectedDataSource.value.data.uniqueDates.length / 7);
});

const maxAllowedDurationMax = computed(() => {
  // Ensures that there's enough data for pre-test (equal to max duration)
  return Math.max(1, Math.floor(fullWeekCount.value / 2));
});

// These computed properties and constants need to be defined before isExplorationParamsValid

const isBudgetSectionDisabled = computed(() => {
  return !selectedDataSource.value?.columns?.costColumn;
});

const isPercentageBudgetDisabled = computed(() => {
  if (isBudgetSectionDisabled.value) return true;
  return selectedDataSource.value?.data?.costSum === 0;
});

const budgetTypeOptions = [
  {
    value: 'percentage_change',
    label: '% Change',
    desc: 'Percentage change relative to Business-As-Usual (BAU) spend.',
  },
  {
    value: 'daily_budget',
    label: '$ Daily',
    desc: 'Incremental daily budget amount (on top of BAU).',
  },
  {
    value: 'total_budget',
    label: '$ Total',
    desc: 'Total incremental budget amount for the experiment duration (on top of BAU).',
  },
];

const filteredBudgetTypeOptions = computed(() => {
  if (isPercentageBudgetDisabled.value) {
    return budgetTypeOptions.filter((opt) => opt.value !== 'percentage_change');
  }
  return budgetTypeOptions;
});

watch(maxAllowedDurationMax, (newMax) => {
  if (parameters.durationMax > newMax) {
    parameters.durationMax = newMax;
  }
  // Ensure min is still less than (now potentially adjusted) max
  if (parameters.durationMin >= parameters.durationMax) {
    parameters.durationMin = Math.max(1, parameters.durationMax - 1);
  }
});

// --- Validation Function and Issues Display ---
// This function MUST be defined AFTER fullWeekCount and maxAllowedDurationMax
function isExplorationParamsValid(): string[] {
  const issues: string[] = [];
  if (!selectedDataSource.value || !selectedDataSource.value.data) {
    issues.push('Data source not fully loaded. Please re-select or wait.');
    return issues;
  }

  // Runtime Durations
  if (parameters.durationMax > maxAllowedDurationMax.value) {
    issues.push(
      `Maximum test duration (${parameters.durationMax} weeks) exceeds the allowable limit of ${maxAllowedDurationMax.value} weeks for the selected data source (which has ${fullWeekCount.value} full weeks). Reduce duration or choose a data source with more historical data.`, // .value is correct here
    );
  }
  if (parameters.durationMin > parameters.durationMax) {
    issues.push('Minimum test duration must be less than the maximum test duration.');
  }

  // Budget Validations
  if (
    isBudgetSectionDisabled.value &&
    parameters.budget.length > 0 &&
    parameters.budget.some((b) => b !== '0' && b !== '')
  ) {
    issues.push('Budget cannot be specified as the selected data source has no cost column.');
  }

  // Metric Validations
  const primaryMetricId = getMetricIdentifier(selectedMetric.value);
  const secondaryMetricIds = secondaryMetrics.value.map(getMetricIdentifier);

  if (secondaryMetricIds.includes(primaryMetricId)) {
    issues.push('The primary metric cannot also be a secondary metric.');
  }

  const uniqueSecondaryMetrics = new Set(secondaryMetricIds);
  if (uniqueSecondaryMetrics.size !== secondaryMetricIds.length) {
    issues.push('Secondary metrics must be unique.');
  }
  console.log(selectedMetric.value);
  validateMetric(selectedMetric.value, issues);
  secondaryMetrics.value.forEach((metric) => {
    validateMetric(metric, issues);
  });
  return issues;
}

function validateMetric(metric: string | AnyMetric, issues: string[]) {
  let columnName;
  if (typeof metric === 'string') {
    columnName = metric;
  } else {
    if (metric.type === 'cpia') {
      if (!metric.conversions_column) {
        issues.push(`Conversion column must be specified for CPiA metric`);
      }
      if (!metric.cost_column) {
        issues.push('Cost column must be specified for CPiA metric');
      }
      return;
    } else if (metric.type === 'iroas') {
      if (!metric.return_column) {
        issues.push(`Return column must be specified for iROAS metric`);
      }
      if (!metric.cost_column) {
        issues.push('Cost column must be specified for iROAS metric');
      }
      return;
    } else if (metric.type === 'custom') {
      if ((metric.cost_per_metric || metric.metric_per_cost) === !!metric.cost_column) {
        issues.push(
          `Metric ${metric.name} has either set cost_per_metric or metric_per_cost flag but no cost column or cost column without one of the flags`,
        );
      }
    }
    columnName = metric.column || metric.name;
  }
  if (columnName) {
    if (!selectedDataSource.value.columns.metricColumns?.includes(columnName)) {
      issues.push('Metric references unknown data source column ' + columnName);
    }
  } else {
    issues.push('Metric has no name or column specified');
  }
}

const explorationValidationIssues = computed(() => {
  // This uses isExplorationParamsValid
  return isExplorationParamsValid();
});

// --- Watchers for dynamic parameter adjustments ---

watch(nCells, (newSize, oldSize) => {
  if (newSize === oldSize) return;

  const currentValues = parameters.cellVolumeConstraint.values;
  const newValues = Array(newSize).fill(null);

  // Copy existing values
  for (let i = 0; i < Math.min(newSize, oldSize); i++) {
    newValues[i] = currentValues[i];
  }
  parameters.cellVolumeConstraint.values = newValues;
});

// This watch needs to be present and after maxAllowedDurationMax
watch(maxAllowedDurationMax, (newMax) => {
  if (parameters.durationMax > newMax) {
    parameters.durationMax = newMax;
  }
  // Ensure min is still less than (now potentially adjusted) max
  if (parameters.durationMin >= parameters.durationMax) {
    parameters.durationMin = Math.max(1, parameters.durationMax - 1);
  }
});

// Validation rule for budget input
const budgetValidationRule = (val: string[]) => {
  if (!val || val.length === 0) {
    return true; // Empty is always allowed (A/B test)
  }

  // Check if all values are numeric
  for (const item of val) {
    if (isNaN(Number(item))) {
      return 'All budget values must be numeric.';
    }
  }

  if (parameters.structure === 'single-cell') {
    return (
      val.length <= 1 || 'For a single-cell test, provide at most one budget value or leave empty.'
    );
  } else {
    // multi-cell
    return (
      val.length === parameters.testGroups ||
      `For ${parameters.testGroups} test groups, provide ${parameters.testGroups} budget values or leave empty.`
    );
  }
};

async function runExploration() {
  let cell_volume_constraint;
  if (parameters.cellVolumeConstraint.enabled) {
    const { type, values, metric_column } = parameters.cellVolumeConstraint;
    cell_volume_constraint = {
      constraint_type: type,
      values: values.map((v) => (v === null || v === '' ? null : Number(v))),
      metric_column: type === 'max_percentage_of_metric' ? metric_column : null,
    };
  }

  const request = {
    // Datasource ID from the selected datasource
    datasource_id: selectedDataSource.value?.id,

    // Core parameters
    primary_metric: selectedMetric.value,
    secondary_metrics: secondaryMetrics.value,

    budgets: parseBudget(parameters),

    // Test parameters
    n_cells: nCells.value,
    alpha: parameters.alpha,
    alternative_hypothesis: parameters.hypothesisType,

    // Duration parameters
    min_runtime_weeks: parameters.durationMin,
    max_runtime_weeks: parameters.durationMax,

    // Methodology options (empty means explore all)
    methodologies: parameters.methodology.length > 0 ? parameters.methodology : [],

    target_power: parameters.power,

    // Geo assignments from user selections
    fixed_geos: getGeoAssignments(geoUnits.value),

    cell_volume_constraint: cell_volume_constraint,

    effect_scope: parameters.effectScope,
    max_trials: Number.isFinite(Number(parameters.maxTrials))
      ? Number(parameters.maxTrials)
      : undefined,
    n_designs: Number.isFinite(Number(parameters.nDesigns))
      ? Number(parameters.nDesigns)
      : undefined,
    // TODO:
    //trimming_quantile_candidates: List[float] = [0.0]
  };
  lastRequest = request;

  const response = await postApiUi<ExperimentExploreResponse>(
    'experiments/explore',
    request,
    'Running exploration',
  );
  if (!response) return;
  // Process the response
  const designs: ExperimentDesign[] = [];
  for (const designResp of response.data.designs) {
    const design: ExperimentDesign = {
      design_id: designResp.design_id,
      mde: designResp.mde,
      runtime_weeks: designResp.runtime_weeks,
      methodology: designResp.methodology,
      methodology_parameters: designResp.methodology_parameters,
      isValid: designResp.evaluation_results?.is_valid_design,
      groups: {
        Control: designResp.geo_assignment?.control || [],
      },
      evaluation_results: designResp.evaluation_results,
      parameters: {
        n_cells: designResp.n_cells,
        primary_metric: designResp.primary_metric,
        secondary_metrics: designResp.secondary_metrics,
        alpha: designResp.alpha,
        alternative_hypothesis: designResp.alternative_hypothesis,
        cell_volume_constraint: designResp.cell_volume_constraint,
        effect_scope: designResp.effect_scope,
        random_seed: designResp.random_seed,
      },
    };
    if (designResp.geo_assignment?.treatment && designResp.geo_assignment.treatment.length > 0) {
      if (designResp.geo_assignment.treatment.length === 1) {
        // This means one treatment group
        design.groups.Treatment = designResp.geo_assignment.treatment[0] || [];
      } else {
        // Multiple treatment groups
        for (let i = 0; i < designResp.geo_assignment.treatment.length; i++) {
          design.groups[`Treatment ${String.fromCharCode(65 + i)}`] =
            designResp.geo_assignment.treatment[i] || [];
        }
      }
    }

    designs.push(design);
  }
  testDesigns.value = designs;
  isExplored.value = true;

  // Navigate to designs tab
  activeTab.value = 'designs';
}

// Helper function to format geo assignments from UI selections
function getGeoAssignments(geoUnits) {
  // Start with empty groups
  const assignments = {
    control: [],
    treatment: [[]], // Start with one treatment group for single-cell
    exclude: [],
  };

  // Process multi-cell test if needed
  if (parameters.structure === 'multi-cell' && parameters.testGroups > 1) {
    // Initialize the right number of treatment groups
    assignments.treatment = Array.from({ length: parameters.testGroups }, () => []);
  }

  // Process each geo unit
  geoUnits.forEach((unit) => {
    if (unit.assignment === 'control') {
      assignments.control.push(unit.geo);
    } else if (unit.assignment === 'exclude') {
      assignments.exclude.push(unit.geo);
    } else if (unit.assignment === 'test') {
      // Default test group (group 0)
      assignments.treatment[0].push(unit.geo);
    } else if (unit.assignment.startsWith('test-')) {
      // Handle test-2, test-3, etc. for multi-cell
      const groupIndex = parseInt(unit.assignment.split('-')[1]) - 1;
      if (groupIndex >= 0 && groupIndex < assignments.treatment.length) {
        assignments.treatment[groupIndex].push(unit.geo);
      }
    }
  });

  return assignments;
}

function parseBudget(parameters: ReturnType<typeof getDefaultParameters>) {
  const budgetValues = parameters.budget;
  // Handle A/B test case (empty budget array) -> return single budget item with value 0
  if (!budgetValues || budgetValues.length === 0) {
    return [{ value: 0, budget_type: parameters.budgetType }];
  }

  const numbers = budgetValues.map((s) => Number(s.trim())).filter((n) => !isNaN(n));

  if (numbers.length !== budgetValues.length) {
    // This case should ideally be caught by input validation, but as a safeguard:
    throw new Error(`Invalid budget values detected during parsing: ${budgetValues.join(',')}`);
  }
  // Map valid numbers to the required API structure
  return numbers.map((i) => {
    return { value: i, budget_type: parameters.budgetType };
  });
}

// Test designs
const isExplored = ref(false);
const testDesigns = ref([] as ExperimentDesign[]);
const sortBy = ref(sortOptions[0]); // Default to Lowest MDE
const sortedDesigns = computed<ExperimentDesign[]>(() => {
  const designs = [...testDesigns.value];
  const field = sortBy.value.value;

  if (field === 'mde') {
    // Sort by MDE, lowest first. Handle undefined mde by pushing them to the end.
    return designs.sort((a, b) => (a.mde ?? Infinity) - (b.mde ?? Infinity));
  } else if (field === 'duration') {
    return designs.sort((a, b) => a.runtime_weeks - b.runtime_weeks);
  }
  // Default return if no specific sort matches, or add more conditions
  return designs;
});

// Design detail dialog
const designDetailDialog = ref(false);
const selectedDesign = ref<ExperimentDesign>(null);
const groupTab = ref('Control');
const geoDetailColumns: QTableColumn[] = [
  { name: 'geo', label: 'Geo Unit', field: (row) => row, align: 'left' },
];

// Format helpers
const formatNumber = (num) => {
  return new Intl.NumberFormat().format(num);
};

const formatKey = (key) => {
  return key
    .replace(/([A-Z])/g, ' $1') // Insert space before capital letters
    .replace(/^./, (str) => str.toUpperCase()); // Capitalize first letter
};

let lastRequest;

function downloadDesign(design) {
  // TODO:
  // Create a design export object
  const exportData = {
    name: `Geo Test Design ${new Date().toISOString().split('T')[0]}`,
    source: selectedDataSource.value?.name,
    metric: getMetricName(selectedMetric.value),
    parameters: design.parameters,
    statistics: {
      mde: design.mde,
      duration: design.runtime_weeks,
    },
    groups: {},
  };

  // Add geo assignments for each group
  Object.keys(design.groups).forEach((groupName) => {
    // design.groups[groupName] is already an array of geo strings
    exportData.groups[groupName] = (design.groups[groupName] as string[]).map((geo: string) => ({
      geo: geo,
    }));
  });
  const filename = `geo-test-design-${Date.now()}.json`;
  exportFile(filename, JSON.stringify(exportData, null, 2));
}

function viewDesign(design) {
  selectedDesign.value = design;
  groupTab.value = 'Control'; // Reset to first tab
  designDetailDialog.value = true;
}

// Helper function to check if a geo was fixed in the assignment
function isFixedGeo(geo: string, groupName: string): boolean {
  if (!lastRequest || !lastRequest.fixed_geos) return false;

  if (groupName === 'Control' && lastRequest.fixed_geos.control?.includes(geo)) {
    return true;
  }

  if (groupName.startsWith('Treatment')) {
    let groupIndex = -1;
    if (groupName === 'Treatment') {
      groupIndex = 0; // Single treatment group
    } else if (groupName.startsWith('Treatment ')) {
      // "Treatment A", "Treatment B"
      const letter = groupName.split(' ')[1];
      if (letter && letter.length === 1) {
        groupIndex = letter.charCodeAt(0) - 65; // 'A' -> 0, 'B' -> 1
      }
    }

    if (
      groupIndex !== -1 && // Valid groupIndex found
      lastRequest.fixed_geos.treatment &&
      lastRequest.fixed_geos.treatment[groupIndex]?.includes(geo)
    ) {
      return true;
    }
  }

  return false;
}

function exportDesignGroupToCsv(design: ExperimentDesign, groupName: string) {
  if (!design || !design.groups || !design.groups[groupName]) {
    $q.notify({
      color: 'negative',
      message: 'Could not find group data to export.',
      icon: 'warning',
    });
    return;
  }

  const geos: string[] = design.groups[groupName];
  if (!geos || geos.length === 0) {
    $q.notify({
      color: 'info',
      message: `Group '${groupName}' has no geo units to export.`,
      icon: 'info',
    });
    return;
  }

  // CSV content: header + data rows
  let csvContent = 'geo_id\n'; // Header
  csvContent += geos.join('\n');

  const designId = design.design_id || 'unknown_design';
  // Sanitize groupName for filename: replace non-alphanumeric (except underscore) with underscore, and convert to lowercase
  const safeGroupName = String(groupName)
    .replace(/[^a-z0-9_]/gi, '_')
    .toLowerCase();
  const filename = `${designId}_${safeGroupName}.csv`;

  const status = exportFile(filename, csvContent, 'text/csv;charset=utf-8;');

  if (status !== true) {
    $q.notify({
      color: 'negative',
      message: 'CSV export failed. Please try again.',
      icon: 'warning',
    });
  } else {
    $q.notify({
      color: 'positive',
      message: `Successfully exported ${filename}`,
      icon: 'check_circle',
    });
  }
}

function getMetricName(metric: AnyMetric | undefined): string {
  if (!metric) return '';
  if (typeof metric === 'string') {
    return metric;
  }
  return metric.name;
}

// Gets a unique identifier for a metric to be used for validation
function getMetricIdentifier(metric: AnyMetric | undefined): string {
  if (!metric) return '';
  if (typeof metric === 'string') {
    return metric;
  }
  if (metric.type === 'custom') {
    return metric.column || metric.name;
  }
  return metric.type;
}

function addSecondaryMetric() {
  if (selectedDataSource.value?.columns?.metricColumns?.length) {
    // Find a metric that is not already in use
    const availableMetrics = selectedDataSource.value.columns.metricColumns.filter(
      (m) =>
        getMetricIdentifier(selectedMetric.value) !== m &&
        !secondaryMetrics.value.map(getMetricIdentifier).includes(m),
    );
    if (availableMetrics.length > 0) {
      secondaryMetrics.value.push(availableMetrics[0]);
    } else {
      $q.notify({
        color: 'warning',
        message: 'No more unique metrics available to add.',
        icon: 'warning',
      });
    }
  }
}

function removeSecondaryMetric(index: number) {
  secondaryMetrics.value.splice(index, 1);
}
</script>
